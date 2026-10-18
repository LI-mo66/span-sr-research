#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
group_dir=${1:?Group directory required}
run_id=${2:?Run ID required}
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'RUNNING\n' > "$group_dir/status.txt"
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED exit=%s\n" "$rc" > "$group_dir/status.txt"; fi' EXIT
"$python_bin" -u research/cata_tab/run.py --config "$group_dir/C1-config.json" --id "$run_id" --screen --screen-updates 50000
"$python_bin" research/cata_tab/compare_reuse.py --audit "$group_dir/reuse-audit.json" --candidate-run "$PWD/runs/$run_id" --output "$group_dir/comparison-50000.json"
printf 'COMPLETE\n' > "$group_dir/status.txt"
if [[ ${AUTO_RELEASE:-0} == 1 ]]; then
    printf 'BACKING_UP\n' > "$group_dir/status.txt"
    "$python_bin" -u research/cata_tab/finalize_reuse.py --project "$PWD" \
        --group-dir "$group_dir" --run-id "$run_id" \
        --destination "${BACKUP_ROOT:?persistent backup directory required}" \
        --receipt "$group_dir/backup-path.txt"
    backup_dir=$(cat "$group_dir/backup-path.txt")
    [[ -f "$backup_dir/BACKUP_COMPLETE" ]] || exit 74
    printf 'COMPLETE; RELEASE_REQUESTED\n' | tee "$group_dir/status.txt" "$backup_dir/release-status.txt"
    sync
    set +e
    featurize instance release < /dev/null
    release_code=$?
    set -e
    printf 'release_command_exit_code=%s\n' "$release_code" >> "$backup_dir/release-status.txt"
    if (( release_code != 0 )); then
        printf 'RELEASE_FAILED; use platform to stop instance.\n' >> "$backup_dir/release-status.txt"
        exit 73
    fi
fi
