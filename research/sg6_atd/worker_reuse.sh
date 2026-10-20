#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
group_dir=${1:?Group directory required}
run_id=${2:?Run ID required}
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'RUNNING\n' > "$group_dir/status.txt"
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED exit=%s\n" "$rc" > "$group_dir/status.txt"; fi' EXIT
"$python_bin" -u research/sg6_atd/run.py --config "$group_dir/A1-config.json" --id "$run_id" --screen --screen-updates 50000
"$python_bin" -u research/sg6_atd/compare_reuse.py --audit "$group_dir/reuse-audit.json" --run "$PWD/runs/$run_id" --output "$group_dir/comparison-50000.json"
printf 'BACKING_UP\n' > "$group_dir/status.txt"
"$python_bin" -u research/sg6_atd/finalize_reuse.py --project "$PWD" --group-dir "$group_dir" --run-id "$run_id" --destination "${BACKUP_ROOT:?}" --receipt "$group_dir/backup-path.txt"
backup=$(cat "$group_dir/backup-path.txt")
[[ -f "$backup/BACKUP_COMPLETE" ]] || exit 74
printf 'COMPLETE; BACKUP_VERIFIED\n' | tee "$group_dir/status.txt" "$backup/release-status.txt"
sync
if [[ ${AUTO_RELEASE:-1} == 1 ]]; then
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$busy" ]]; then
        printf 'COMPLETE; RELEASE_SKIPPED_GPU_BUSY: %s\n' "$busy" | tee "$group_dir/status.txt" "$backup/release-status.txt"
        exit 0
    fi
    printf 'COMPLETE; RELEASE_REQUESTED\n' | tee "$group_dir/status.txt" "$backup/release-status.txt"
    sync
    set +e
    featurize instance release < /dev/null
    rc=$?
    set -e
    printf 'release_command_exit_code=%s\n' "$rc" >> "$backup/release-status.txt"
    if (( rc != 0 )); then
        printf 'RELEASE_FAILED\n' >> "$backup/release-status.txt"
        exit 73
    fi
fi
