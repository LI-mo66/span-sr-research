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
    if [[ ${AUTO_RELEASE:-0} == 1 ]]; then
        set +e
        backup_root=${BACKUP_ROOT:?persistent backup directory required}
        group=${MOE_FS_GROUP:?group required}
        mkdir -p "$backup_root"
        durable_status="$backup_root/${group}-$(basename "$job")-status.txt"
        cp "$job/status.txt" "$durable_status"
        "$python_bin" -u research/moe_fs/finalize.py --project "$PWD" --group "$group" \
            --destination "$backup_root" --training-exit-code "$code"
        backup_code=$?
        printf 'backup_exit_code=%s\n' "$backup_code" | tee -a "$job/status.txt" "$durable_status"
        if [[ "$backup_code" != 0 ]]; then
            printf 'BACKUP_FAILED; instance retained. Back up manually before release.\n' | tee -a "$job/status.txt" "$durable_status"
            exit 74
        fi
        printf 'release_requested=true\n' | tee -a "$job/status.txt" "$durable_status"
        sync
        featurize instance release < /dev/null
        release_code=$?
        printf 'release_command_exit_code=%s\n' "$release_code" >> "$durable_status"
        if [[ "$release_code" != 0 ]]; then
            printf 'RELEASE_FAILED; stop instance through platform.\n' >> "$durable_status"
            exit 73
        fi
    fi
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_bin" -u research/moe_fs/screen.py "$@"
