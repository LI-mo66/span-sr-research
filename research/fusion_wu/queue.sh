#!/usr/bin/env bash
# Run F1 then F2; never restart completed/failed runs automatically.
set -Eeuo pipefail
first=${1:?F1 project required}
second=${2:?F2 project required}
receipt=${3:?Queue output directory required}
[[ ${AUTO_RELEASE:-1} == 0 || ${AUTO_RELEASE:-1} == 1 ]]
mkdir -p "$receipt"
trap 'rc=$?; if ((rc!=0)); then echo "FAILED exit=$rc; INSTANCE_RETAINED" > "$receipt/status.txt"; fi' EXIT
for arm in F1 F2; do
    if [[ "$arm" == F1 ]]; then project=$first; else project=$second; fi
    [[ "$project" == /home/featurize/work/span-x4-research/code/* ]]
    [[ -d "$project" && ! -e "$project/logs/fusion-WU-launch" && ! -e "$project/logs/fusion-WU" ]]
    manager="$project/logs/fusion-WU-launch"
    mkdir -p "$manager"
    printf '%s\n' "RUNNING_$arm" | tee "$receipt/status.txt"
    bash "$project/research/fusion_wu/worker_reuse.sh" "$manager" "$arm" > "$manager/nohup.log" 2>&1 &
    pid=$!;printf '%s\n' "$pid" > "$manager/worker.pid"
    wait "$pid" || { tail -n 70 "$manager/nohup.log"; exit 1; }
    grep -qx COMPLETE_NO_SHUTDOWN "$manager/status.txt"
    /environment/miniconda3/bin/python3 - "$project" "$receipt" "$arm" <<'PY'
import hashlib,json,sys
from pathlib import Path
project,receipt=map(Path,sys.argv[1:3]);arm=sys.argv[3]
backup=Path((project/'logs/fusion-WU/backup-path.txt').read_text().strip()).resolve()
assert backup.is_relative_to(Path('/home/featurize/work/span-x4-research/records'))
assert (backup/'BACKUP_COMPLETE').is_file()
manifest=json.loads((backup/'backup-manifest.json').read_text());assert manifest['training_exit_code']==0
for row in manifest['files']:
    path=backup/row['path'];assert path.resolve().is_relative_to(backup) and not path.is_symlink()
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    assert h.hexdigest()==row['sha256'],str(path)
result=json.loads((backup/'logs/comparison.json').read_text())
(receipt/(arm+'-comparison.json')).write_text(json.dumps(result,indent=2))
(receipt/(arm+'-backup.txt')).write_text(str(backup)+'\n')
print(arm,'COMPLETE_BACKUP_VERIFIED',backup,flush=True)
PY
done
printf 'COMPLETE_BOTH_BACKUPS_VERIFIED\n' | tee "$receipt/status.txt"
sync
if [[ ${AUTO_RELEASE:-1} == 1 ]]; then
    exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
    flock -n 9 || { echo RELEASE_SKIPPED_RESEARCH_LOCK_BUSY; exit 0; }
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$busy" ]]; then echo "RELEASE_SKIPPED_GPU_BUSY: $busy"; exit 0; fi
    echo RELEASE_REQUESTED | tee -a "$receipt/status.txt"
    featurize instance release < /dev/null
fi
