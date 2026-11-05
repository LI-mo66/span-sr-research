#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset MS_VERIFY_CASE
manager=logs/multiscale-manager
phase=STARTING
trap 'rc=$?; if ((rc!=0)); then printf "FAILED phase=%s exit=%s; INSTANCE_RETAINED\n" "$phase" "$rc" > "$manager/status.txt"; fi' EXIT
[[ -z $(git status --porcelain) ]]
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY;exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU_BUSY: $busy";exit 75; }
stage() { phase=$1; echo "$phase" | tee "$manager/status.txt"; }
stage PREPARING_DATA_AND_CONFIGS
"$python_bin" -u research/multiscale_f0/prepare.py
mkdir -p /home/featurize/span-multiscale-runtime/image_cache datasets
"$python_bin" - <<'PY'
from pathlib import Path
p=Path('datasets/image_cache');target=Path('/home/featurize/span-multiscale-runtime/image_cache')
if p.is_symlink():assert p.resolve()==target.resolve()
elif p.exists():raise RuntimeError('Unexpected existing cache; refusing overwrite')
else:p.symlink_to(target,target_is_directory=True)
PY
"$python_bin" -m pip freeze > logs/multiscale50k/dependencies.txt
stage VERIFYING_MODELS_CROPS_X8_AND_EXACT_RESUME
"$python_bin" -u research/multiscale_f0/verify_x4.py
"$python_bin" -u research/multiscale_f0/verify.py
# Check all four real micro16/accum4 and largest-image paths before spending50K.
for scale in 2 3; do
  for arm in B0 F0; do
    stage "REAL_SMOKE_X${scale}_${arm}"
    "$python_bin" -u research/multiscale_f0/smoke.py "logs/multiscale50k/x${scale}-${arm}.json" "logs/multiscale50k/smoke-x${scale}-${arm}.json"
  done
done
echo PASS > logs/multiscale50k/PREFLIGHT_COMPLETE
if [[ ${CHECK_ONLY:-0} == 1 ]]; then stage PREFLIGHT_COMPLETE_NO_TRAINING;exit 0;fi
for scale in 2 3; do
  for arm in B0 F0; do
    id="MS-X${scale}-${arm}-S10-50K-01"
    stage "TRAINING_$id"
    "$python_bin" -u research/multiscale_f0/run.py --config "logs/multiscale50k/x${scale}-${arm}.json" --id "$id" --until-update 50000
    if [[ "$arm" == F0 ]]; then "$python_bin" -u research/multiscale_f0/finish.py "$scale";fi
    stage "BACKING_UP_$id"
    "$python_bin" -u research/multiscale_f0/backup.py "$id"
  done
done
stage COMPLETE_FOUR_SCREENS_BACKUPS_VERIFIED_NO_SHUTDOWN
