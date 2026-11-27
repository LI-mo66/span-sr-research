#!/usr/bin/env bash
set -Eeuo pipefail
project=$1
source_project=$2
output=$3
cache=$4
cd "$project"
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE
phase=CHECKING_GPU
trap 'rc=$?; if ((rc!=0)); then printf "FAILED phase=%s exit=%s INSTANCE_RETAINED\n" "$phase" "$rc" > "$output/status.txt"; fi' EXIT
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
[[ -z $(git status --porcelain) ]]
phase=VERIFYING_THEN_REPLAYING_295K
echo "$phase" > "$output/status.txt"
"$python_bin" -u research/f0_x3_update_audit/verify.py --device cuda
"$python_bin" -u research/f0_x3_update_audit/audit.py \
    --source-project "$source_project" --output "$output" --cache "$cache"
echo DIAGNOSTIC_COMPLETE_READ_REPORT_NO_SHUTDOWN > "$output/status.txt"
