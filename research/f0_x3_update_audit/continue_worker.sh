#!/usr/bin/env bash
set -Eeuo pipefail
project=$1
source_project=$2
stage=$3
cache=$4
manager=$5
cd "$project"
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE
phase=CHECKING_GPU
trap 'rc=$?; if ((rc!=0)); then printf "FAILED phase=%s exit=%s INSTANCE_RETAINED\n" "$phase" "$rc" > "$stage/status.txt"; fi' EXIT
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
[[ $(git rev-parse HEAD) == 6e71b81cdcb53479233b674d35132f7233f96e64 ]]
[[ -z $(git status --porcelain) ]]
phase=STAGED_FRONTCAP_CONTINUATION_WITH_QUALITY_GATES
echo "$phase" > "$stage/status.txt"
"$python_bin" -u "$manager" --project "$project" --source-project "$source_project" \
    --stage "$stage" --cache "$cache"
