#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
cap=$1
manager=$2
python_bin=/environment/miniconda3/bin/python3
id=FEAT-SG6-F0-X8-S10-1M-01
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
phase=STARTING
stage() { phase=$1; printf '%s\n' "$phase" | tee "$manager/status.txt"; }
trap 'rc=$?; if ((rc!=0)); then echo "FAILED phase=$phase exit=$rc; INSTANCE_RETAINED" > "$manager/status.txt"; fi' EXIT
[[ ${AUTO_RELEASE:-1} == 0 || ${AUTO_RELEASE:-1} == 1 ]]
if [[ ${AUTO_RELEASE:-1} == 1 ]]; then command -v featurize >/dev/null; fi
[[ -z $(git status --porcelain) ]]
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY;exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU_BUSY: $busy";exit 75; }
mkdir -p /home/featurize/span-f0-long-runtime/image_cache datasets
if [[ ! -e datasets/image_cache && ! -L datasets/image_cache ]]; then
    ln -s /home/featurize/span-f0-long-runtime/image_cache datasets/image_cache
fi
[[ $(readlink -f datasets/image_cache) == /home/featurize/span-f0-long-runtime/image_cache ]]
resume=()
if [[ ! -e logs/f0-long/frozen/config.json ]]; then
    [[ "$cap" == 100000 && ! -e "runs/$id" ]]
    stage RESTORING_DATA_PATHS
    "$python_bin" -u research/f0_long/data_paths.py
    stage AUDITING_FIXED_F0_DF2K_AND_SIX_DATASETS
    "$python_bin" -u research/f0_long/prepare.py \
        --wrun /home/featurize/work/span-x4-research/code/span-sg6-mid-window-50k-auto/runs/FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01-A-s10 \
        --urun /home/featurize/work/span-x4-research/records/FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01-20261025T125002Z/runs/FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01
    git archive --format=tar --output=logs/f0-long/source.tar HEAD
    git bundle create logs/f0-long/source.bundle HEAD
    "$python_bin" -m pip freeze > logs/f0-long/dependencies.txt
    stage VERIFYING_F0_CPU_AND_CUDA
    "$python_bin" -u research/fusion_wu/verify.py cpu
    "$python_bin" -u research/fusion_wu/verify.py cuda
    stage VERIFYING_X8_AND_EXACT_STAGE_CONTINUATION
    "$python_bin" -u research/f0_long/test_protocol.py > logs/f0-long/test-protocol-server.log 2>&1 || {
        tail -n 80 logs/f0-long/test-protocol-server.log;exit 1;
    }
    "$python_bin" research/f0_long/guard.py
    stage REAL_MICRO16_ACCUM4_CHECK
    "$python_bin" -u research/fusion_wu/smoke.py --config logs/fusion-WU/frozen/config.json --output logs/f0-long/real-smoke.json
    stage REPRODUCING_ARCHIVED_48K_AND_MEASURING_X8
    "$python_bin" -u research/f0_long/server_check.py
    printf 'All native engineering and historical reproduction checks passed.\n' > logs/f0-long/PREFLIGHT_COMPLETE
else
    [[ -f logs/f0-long/PREFLIGHT_COMPLETE ]]
    if [[ -f "runs/$id/latest.pt" ]]; then resume=(--resume)
    else [[ "$cap" == 100000 && ! -e "runs/$id" ]];fi
fi
if [[ ${CHECK_ONLY:-0} == 1 ]]; then stage NATIVE_PREFLIGHT_COMPLETE_NO_FORMAL_TRAINING;exit 0;fi
stage CHECKING_STAGE_STORAGE
"$python_bin" - "$cap" <<'PY'
import json,shutil,sys
from pathlib import Path
cap=int(sys.argv[1]);size=json.loads(Path('logs/f0-long/real-smoke.json').read_text())['estimated_checkpoint_bytes']
required=2*(cap//1000+4)*size+2*1024**3
free=shutil.disk_usage('.').free
print('Conservative free-space check, bytes:',required,'available:',free,flush=True)
assert free>required,'Expand storage before this stage; no training started'
PY
stage "TRAINING_TO_TOTAL_$cap"
"$python_bin" -u research/f0_long/run.py --config logs/f0-long/frozen/config.json \
    --id "$id" --until-update "$cap" "${resume[@]}"
stage ARCHIVING_STAGE
"$python_bin" -u research/f0_long/archive.py "$cap"
stage STAGE_COMPLETE_BACKUP_VERIFIED
sync
if [[ ${AUTO_RELEASE:-1} == 1 ]]; then
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$busy" ]]; then echo RELEASE_SKIPPED_GPU_BUSY;exit 0;fi
    echo RELEASE_REQUESTED | tee -a "$manager/status.txt"
    featurize instance release < /dev/null
fi
