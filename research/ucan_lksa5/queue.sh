#!/usr/bin/env bash
# One GPU, six independent immutable checkouts. All preflights before training.
set -Eeuo pipefail
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
manifest=${1:?manifest.tsv required}
queue=${2:?persistent queue directory required}
reference=${3:?SG6 archive required}
context=${4:?independent p24 archive required}
records=${5:?persistent records root required}
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo 'Shared GPU lock held; no training or release'; exit 75; }
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED exit=%s; instance retained\n" "$rc" | tee "$queue/FAILED.txt"; fi' EXIT
[[ $(wc -l < "$manifest") == 6 ]] || { echo 'Need exactly six arms'; exit 1; }
for phase in preflight train; do
  while IFS=$'\t' read -r arm commit project directory run_id; do
    [[ $(git -C "$project" rev-parse HEAD) == "$commit" ]]
    printf '\nPHASE=%s ARM=%s ID=%s\n' "$phase" "$arm" "$run_id"
    "$python_bin" -u "$project/$directory/controller.py" --phase "$phase" --id "$run_id" \
      --reference "$reference" --context-reference "$context" --records "$records" \
      > "$queue/$arm-$phase.log" 2>&1
    echo "Completed $phase $arm; log=$queue/$arm-$phase.log"
  done < "$manifest"
done
# Children and CUDA contexts have exited. Rehash all six persistent archives.
while IFS=$'\t' read -r arm commit project directory run_id; do
  "$python_bin" "$project/$directory/release_guard.py" "$project/logs/$(basename "$directory")/$run_id/status.json" \
    >> "$queue/verified-backups.txt"
done < "$manifest"
"$python_bin" - "$manifest" "$queue" <<'PY'
import hashlib,json,sys,zipfile
from pathlib import Path
manifest=Path(sys.argv[1]);queue=Path(sys.argv[2]);summary={}
with zipfile.ZipFile(queue/'queue-results.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for line in manifest.read_text().splitlines():
        arm,commit,project,directory,rid=line.split('\t')
        status=json.loads((Path(project)/'logs'/Path(directory).name/rid/'status.json').read_text())
        assert status['success'] and status['exit_code']==0
        archive=Path(status['backup']);summary[arm]={'commit':commit,'backup':str(archive),'comparison':json.loads((archive/'logs/frozen/comparison.json').read_text())}
        for path in archive.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.json','.jsonl','.md','.txt','.png','.csv'):
                z.write(path,arm+'/'+path.relative_to(archive).as_posix())
    payload=json.dumps(summary,indent=2);z.writestr('QUEUE_SUMMARY.json',payload)
(queue/'QUEUE_SUMMARY.json').write_text(payload)
with zipfile.ZipFile(queue/'queue-results.zip') as z:assert z.testzip() is None
sha=hashlib.sha256((queue/'queue-results.zip').read_bytes()).hexdigest()
(queue/'queue-results.zip.sha256').write_text(sha+'  queue-results.zip\n')
(queue/'SUCCESS.json').write_text(json.dumps({'success':True,'verified_archives':6,'results_sha256':sha},indent=2))
PY
sync
if [[ ${AUTO_RELEASE:-1} != 1 ]]; then echo 'Six runs complete; automatic release disabled'; exit 0; fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$busy" ]]; then echo 'RELEASE_SKIPPED_GPU_BUSY' | tee "$queue/release-status.txt"; exit 0; fi
echo 'RELEASE_REQUESTED_AFTER_SIX_VERIFIED_BACKUPS' | tee "$queue/release-status.txt"
sync
featurize instance release < /dev/null
echo 'release_command_exit_code=0' >> "$queue/release-status.txt"
