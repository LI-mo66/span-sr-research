#!/usr/bin/env bash
set -Eeuo pipefail
project=$1
stage=$2
python_bin=$3
cd "$project"

trap 'rc=$?; if ((rc != 0)); then
    phase=$(cat "$stage/status.txt" 2>/dev/null || true)
    printf "FAILED exit=%s phase=%s; INSTANCE_RETAINED\n" "$rc" "$phase" |
        tee "$stage/status.txt"
fi' EXIT

echo CHECKING_GPU_LOCK > "$stage/status.txt"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE

mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { printf 'GPU_BUSY: %s\n' "$busy"; exit 75; }

[[ $(git rev-parse HEAD) == 0748bd60c1eaddda4e0db7f01ba768a0f2a22be3 ]]
[[ -z $(git status --porcelain) ]]
"$python_bin" -u "$stage/continue.py"
