#!/usr/bin/env bash
set -euo pipefail
job=${1:?job path required}
shift
python_bin=${PYTHON_BIN:-python}
exec 9>logs/server-gpu.lock
if ! flock -n 9; then
    printf 'exit_code=75\nreason=GPU_queue_lock_held\n' > "$job/status.txt"
    exit 75
fi
finish() {
    code=$?
    trap - EXIT
    printf 'exit_code=%s\nfinished_utc=%s\n' "$code" "$(date -u +%FT%TZ)" > "$job/status.txt"
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_bin" -u research/moe_fs/screen.py "$@"
