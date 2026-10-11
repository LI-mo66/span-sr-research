#!/usr/bin/env bash
set -euo pipefail
job=$1
shift
python_bin=${PYTHON_BIN:-python}
shutdown_bin=${SHUTDOWN_BIN:-/usr/bin/shutdown}
finish() {
    code=$?
    trap - EXIT
    printf 'exit_code=%s\nfinished_utc=%s\n' "$code" "$(date -u +%FT%TZ)" > "$job/status.txt"
    sync
    printf 'Queue exited %s; invoking shutdown: %s\n' "$code" "$shutdown_bin"
    if ! "$shutdown_bin"; then
        printf 'SHUTDOWN_FAILED: use platform console to power off.\n' | tee -a "$job/status.txt"
    fi
    exit "$code"
}
# An unowned GPU lock must not cause this process to power off another queue.
exec 9>logs/server-gpu.lock
if ! flock -n 9; then
    printf 'GPU queue lock already held; refusing launch without shutdown.\n' > "$job/status.txt"
    exit 75
fi
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_bin" -u research/server_queue.py "$@"
