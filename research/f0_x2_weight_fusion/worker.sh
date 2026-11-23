#!/usr/bin/env bash
set -Eeuo pipefail
project=$1
source_run=$2
finetune=$3
output=$4
cd "$project"
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE
phase=PREFLIGHT
trap 'rc=$?; if ((rc!=0)); then printf "FAILED phase=%s exit=%s INSTANCE_RETAINED\n" "$phase" "$rc" > "$output/status.txt"; fi' EXIT
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
[[ -z $(git status --porcelain) ]]
auto_release=${AUTO_RELEASE:-0}
[[ "$auto_release" == 0 || "$auto_release" == 1 ]]
if [[ "$auto_release" == 1 ]]; then
    release_cli=$(command -v featurize)
    "$release_cli" instance release --help > "$output/release-help.txt"
fi
phase=VERIFYING_AND_EVALUATING_SEVEN_VARIANTS
echo "$phase" > "$output/status.txt"
"$python_bin" -u research/f0_x2_weight_fusion/evaluate.py \
    --source-run "$source_run" --finetune-project "$finetune" --output "$output"
test -s "$output/COMPLETE.json"
echo COMPLETE_WEIGHT_FUSION_EVALUATION > "$output/status.txt"
if [[ "$auto_release" == 1 ]]; then
    phase=CHECKING_GPU_BEFORE_RELEASE
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
    phase=RELEASING_INSTANCE
    echo "$phase" > "$output/status.txt"
    "$release_cli" instance release < /dev/null
fi
