#!/usr/bin/env bash
set -Eeuo pipefail
manager=$1
export FUSION_ARM=${2:?F1 or F2 required}
[[ "$FUSION_ARM" == F1 || "$FUSION_ARM" == F2 ]]
repo=/home/featurize/work/span-x4-research/code/span-sr-research
project=$(cd "$(dirname "$0")/../.." && pwd)
branch=research/sg6-wu-fusion-v1
base=e731483d4ce4e325f3d64dbbc80add342b46f7f1
wcommit=78cc1dda90a5bafcefa6c62225197f6eb95c92e2
ucommit=d42b73fef5b2ec3e088f340ab033a5c4ac13a190
python_bin=/environment/miniconda3/bin/python3
wrun=/home/featurize/work/span-x4-research/code/span-sg6-mid-window-50k-auto/runs/FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01-A-s10
urun=/home/featurize/work/span-x4-research/records/FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01-20261025T125002Z/runs/FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01
runid="FEAT-SG6-WU-LKSA3-$FUSION_ARM-50K-01"
phase=STARTING
stage() { phase=$1; printf '%s\n' "$phase" | tee "$manager/status.txt"; }
trap 'rc=$?; if ((rc!=0)); then printf "FAILED phase=%s exit=%s; NO_SHUTDOWN\n" "$phase" "$rc" > "$manager/status.txt"; fi' EXIT
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
command -v git >/dev/null
command -v flock >/dev/null
test -f "$wrun/manifest.json"
test -f "$urun/manifest.json"
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo 'GPU research lock busy; stopped'; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$busy" || { echo "GPU busy: $busy; stopped"; exit 75; }

stage CHECKING_ISOLATED_COMMITTED_SOURCE
cd "$project"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "$base" HEAD
for commit in "$wcommit" "$ucommit"; do
    git cat-file -e "$commit^{commit}"
done

stage PREPARING_DATA_PATHS_AND_CACHE
"$python_bin" - "$project" <<'PY'
import sys
from pathlib import Path
project=Path(sys.argv[1]);root=Path('/home/featurize/datasets');p=root/'Flickr2K'
def valid(p):return (p/'Flickr2K_HR').is_dir() and (p/'Flickr2K_LR_bicubic/X4').is_dir()
if not valid(p):
    if p.exists() or p.is_symlink():raise RuntimeError('Flickr2K path incomplete; refusing overwrite')
    other=root/'downloads/Flickr2K'
    if valid(other):p.symlink_to(other,target_is_directory=True)
    else:
        assert (root/'downloads/Flickr2K.tar').is_file(),'Missing Flickr2K archive; no training started'
        sys.path.insert(0,str(project/'research'));import extract_flickr2k
        extract_flickr2k.ROOT=Path('/home/featurize');sys.argv=['extract_flickr2k.py'];extract_flickr2k.main()
assert valid(p)
cache=Path('/home/featurize/span-sg6-wu-orders-runtime/image_cache');cache.mkdir(parents=True,exist_ok=True)
link=project/'datasets/image_cache';link.parent.mkdir(parents=True,exist_ok=True)
assert not link.exists() and not link.is_symlink()
link.symlink_to(cache,target_is_directory=True)
PY
stage AUDITING_SG6_W_U_PROTOCOL_DATA_ENVIRONMENT
"$python_bin" -u research/fusion_wu/workflow.py prepare --wrun "$wrun" --urun "$urun"
"$python_bin" -m pip freeze > logs/fusion-WU/dependencies.txt
git archive --format=tar --output=logs/fusion-WU/source.tar HEAD
git bundle create logs/fusion-WU/source.bundle HEAD

stage VERIFYING_CPU_INTERFACE_AND_EXACT_REFERENCE_FORWARD
"$python_bin" -u research/fusion_wu/verify.py cpu
stage VERIFYING_CUDA_INTERFACE_AND_UPDATES
"$python_bin" -u research/fusion_wu/verify.py cuda
stage VERIFYING_SHARED_TRAINER_EXACT_RESUME
"$python_bin" -u research/fusion_wu/integration.py > logs/fusion-WU/integration.log 2>&1 || {
    tail -n 70 logs/fusion-WU/integration.log; exit 1;
}
cat logs/fusion-WU/integration.json
stage CHECKING_REAL_MICRO16_ACCUM4_AND_LARGEST_IMAGE
"$python_bin" -u research/fusion_wu/smoke.py --config logs/fusion-WU/frozen/config.json --output logs/fusion-WU/real-smoke.json
stage BENCHMARKING_SG6_W_U_FUSION
"$python_bin" -u research/fusion_wu/benchmark.py
stage CHECKING_DISK_FOR_TRAINING_AND_BACKUP
"$python_bin" - <<'PY'
import json,shutil
from pathlib import Path
smoke=json.loads(Path('logs/fusion-WU/real-smoke.json').read_text())
required=2*52*smoke['estimated_checkpoint_bytes']+2*1024**3
free=shutil.disk_usage('.').free
print('Required free bytes:',required,'Available:',free,flush=True)
assert free>required,'Insufficient space for run plus verified backup'
PY
test -z "$(git status --porcelain)"
stage TRAINING_FUSION_FROM_SCRATCH_TO_50000
"$python_bin" -u research/fusion_wu/run.py --config logs/fusion-WU/frozen/config.json \
    --id "$runid" --screen --screen-updates 50000
stage COMPARING_AND_BACKING_UP
"$python_bin" -u research/fusion_wu/workflow.py finish
stage COMPLETE_NO_SHUTDOWN
echo 'Fusion screen and SHA256 backup complete. Automatic shutdown is disabled.'