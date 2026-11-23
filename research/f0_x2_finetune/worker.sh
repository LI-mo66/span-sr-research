#!/usr/bin/env bash
set -Eeuo pipefail
cd "$1"
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
source_run=$2
group=$3
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE
phase=CHECKING_GPU
trap 'rc=$?; if ((rc!=0)); then printf "FAILED phase=%s exit=%s\n" "$phase" "$rc" > "$group/status.txt"; fi' EXIT
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU_BUSY: $busy"; exit 75; }
[[ -z $(git status --porcelain) ]]
auto_release=${AUTO_RELEASE:-0}
[[ "$auto_release" == 0 || "$auto_release" == 1 ]]
if [[ "$auto_release" == 1 ]]; then
    release_cli=$(command -v featurize)
    "$release_cli" instance release --help > "$group/release-help.txt"
fi
phase=AUDITING_982K_AND_DATA
echo "$phase" > "$group/status.txt"
"$python_bin" -u research/f0_x2_finetune/run.py --source-run "$source_run" --group "$group" --preflight
phase=VERIFYING_ENGINEERING
echo "$phase" > "$group/status.txt"
"$python_bin" -u research/f0_x2_finetune/verify.py --source-run "$source_run"
for arm in l1_lr5 mse_lr5; do
    phase="TRAINING_$arm"
    echo "$phase" > "$group/status.txt"
    "$python_bin" -u research/f0_x2_finetune/run.py --source-run "$source_run" --group "$group" --arm "$arm"
    "$python_bin" research/f0_x2_finetune/finish.py "$group" > "$group/comparison.log"
done
phase=VERIFYING_AND_FLUSHING_RESULTS
echo "$phase" > "$group/status.txt"
CUDA_VISIBLE_DEVICES="" "$python_bin" research/f0_x2_finetune/finalize.py "$group"
echo COMPLETE_TWO_PATCH96_FINETUNES > "$group/status.txt"
if [[ "$auto_release" == 1 ]]; then
    phase=CHECKING_GPU_BEFORE_RELEASE
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$busy" ]] || { echo "GPU_BUSY: $busy"; exit 75; }
    phase=RELEASING_INSTANCE
    echo "$phase" > "$group/status.txt"
    "$release_cli" instance release < /dev/null
fi
