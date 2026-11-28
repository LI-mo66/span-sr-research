#!/usr/bin/env bash
set -Eeuo pipefail
manager_commit=$1
repo=/home/featurize/work/span-x4-research/code/span-f0-x2x3-screen-01
project=/home/featurize/work/span-x4-research/code/span-f0-x3-frontcap-296k-04
stage="$project/logs/f0_x3_update_audit/FRONTCAP-310K-400K-07"
cache=/home/featurize/span-f0-x3-audit-runtime/image_cache
python_bin=/environment/miniconda3/bin/python3
training_commit=6e71b81cdcb53479233b674d35132f7233f96e64

test -d "$project"
mkdir -p "$project/logs"
bootlog="$project/logs/launch-to400-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$bootlog") 2>&1
trap 'rc=$?; echo "LAUNCH_FAILED line=$LINENO exit=$rc log=$bootlog"; exit "$rc"' ERR

test -x "$python_bin"
command -v flock >/dev/null
command -v nvidia-smi >/dev/null
[[ ! -e "$stage" ]] || { echo "STAGE_ALREADY_EXISTS: $stage"; exit 1; }
[[ $(git -C "$project" rev-parse HEAD) == "$training_commit" ]]
[[ -z $(git -C "$project" status --porcelain) ]]

CUDA_VISIBLE_DEVICES="" "$python_bin" - "$project" "$repo" <<'PY'
import json,shutil,sys
from pathlib import Path
project,source=map(Path,sys.argv[1:])
run=project/'runs/F0-X3-FRONTCAP5E4-S10-V1'
assert not (run/'ANOMALY.json').exists(), 'ANOMALY_PRESENT'
for name in ('latest.pt','COMPLETE-310000.json','verification-310000.json'):
    assert (run/name).is_file(), name
report=json.loads((project/'logs/f0_x3_update_audit/FRONTCAP-300K-310K-06/report.json').read_text())
assert report['status']=='COMPLETE_310K_REVIEW_REQUIRED_NO_SHUTDOWN'
assert report['update']==310000 and report['stop_reason'] is None
original=source/'runs/F0-X3-CLIP01-U300K-14'
assert (original/'epoch00295_u295000.pt').is_file()
assert (original/'loss_steps.jsonl').is_file()
manifest=json.loads((original/'manifest.json').read_text())
assert Path(manifest['config']['paths_file']).is_file()
missing=[str(row[key]) for group in manifest['data'].values() for row in group for key in ('hr','lr') if not Path(row[key]).is_file()]
if missing:
    print('MISSING_ORIGINAL_IMAGES',len(missing),*missing[:30],sep='\n')
    raise SystemExit(1)
# Budget90 additional checkpoints plus safety margin; never delete existing files.
required=int(90*(run/'latest.pt').stat().st_size*1.3)+1024**3
free=shutil.disk_usage(run).free
print(f'Disk free={free/1024**3:.2f}GiB, estimated additional requirement={required/1024**3:.2f}GiB')
assert free>=max(required,2*1024**3), 'INSUFFICIENT_PERSISTENT_SPACE'
print('PASS: prior310K completion, original data paths, disk preflight',flush=True)
PY

if [[ -L "$cache" && ! -e "$cache" ]]; then
    echo "BROKEN_CACHE_SYMLINK: $cache"; exit 1
fi
mkdir -p "$cache"
test -w "$cache"
git -C "$repo" cat-file -e "$manager_commit^{commit}"
local_dir=$(mktemp -d /tmp/span-x3-to400-XXXXXXXX)
git -C "$repo" show "$manager_commit:research/f0_x3_update_audit/continue_to400.py" > "$local_dir/manager.py"
git -C "$repo" show "$manager_commit:research/f0_x3_update_audit/continue_worker.sh" > "$local_dir/worker.sh"
"$python_bin" -m py_compile "$local_dir/manager.py"
bash -n "$local_dir/worker.sh"
mkdir -p "$(dirname "$stage")"
mkdir "$stage"
cp "$local_dir/manager.py" "$stage/manager.py"
cp "$local_dir/worker.sh" "$stage/worker.sh"
export PYTHON_BIN="$python_bin" PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nohup bash "$local_dir/worker.sh" "$project" "$repo" "$stage" "$cache" "$local_dir/manager.py" > "$stage/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$stage/pid.txt"
printf 'SUBMITTED PID=%s\nLOG=%s/nohup.log\nBOOTLOG=%s\n' "$pid" "$stage" "$bootlog"
echo '310K to400K; 1K stages; original LR milestones; no automatic shutdown.'
