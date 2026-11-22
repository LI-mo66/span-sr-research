#!/usr/bin/env bash
set -Eeuo pipefail
cd "$1"
stage=$2
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE
trap 'rc=$?; if ((rc!=0)); then
    printf "FAILED exit=%s; READ_NOHUP_LOG\n" "$rc" > "$stage/status.txt"
fi' EXIT

mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU_BUSY: $busy"; exit 75; }
[[ $(git rev-parse HEAD) == 0748bd60c1eaddda4e0db7f01ba768a0f2a22be3 ]]
[[ -z $(git status --porcelain) ]]
/environment/miniconda3/bin/python3 -u "$stage/manager.py"
