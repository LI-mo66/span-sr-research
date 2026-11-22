#!/usr/bin/env bash
set -Eeuo pipefail

cd "$1"
stage=$2
python_bin=$3
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE

phase=PREFLIGHT
trap 'rc=$?; if ((rc != 0)); then
    printf "FAILED phase=%s exit=%s\n" "$phase" "$rc" | tee "$stage/status.txt"
fi' EXIT

printf '%s\n' "$phase" > "$stage/status.txt"
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { printf 'GPU_BUSY: %s\n' "$busy"; exit 75; }

test "$(git rev-parse HEAD)" = 0748bd60c1eaddda4e0db7f01ba768a0f2a22be3
test -z "$(git status --porcelain)"
test -d datasets/image_cache
test "$(cat logs/x3-clip01-210k-300k-14/status.txt)" = COMPLETE_X3_CLIP01_300K_NO_SHUTDOWN

CUDA_VISIBLE_DEVICES="" "$python_bin" "$stage/prepare.py" "$stage"

phase=TRAINING_X3_RECOVERY_295K_TO_300K
printf '%s\n' "$phase" > "$stage/status.txt"
"$python_bin" -u - "$stage/continue.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
code = compile(path.read_bytes(), str(path), "exec")
sys.argv = [str(path)]
exec(code, {"__name__": "__main__", "__file__": str(path)})
PY

phase=VERIFYING_300K
CUDA_VISIBLE_DEVICES="" "$python_bin" "$stage/check.py" "$stage"
echo COMPLETE_X3_CLIP05_TEST_300K_NO_SHUTDOWN > "$stage/status.txt"
