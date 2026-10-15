#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
group_dir=${1:?Group directory required}
run_id=${2:?Run ID required}
resume=${3:-}
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'RUNNING_A6_ONLY\n' > "$group_dir/status.txt"
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED exit=%s; instance retained\n" "$rc" > "$group_dir/status.txt"; fi' EXIT
extra=()
[[ "$resume" != resume ]] || extra+=(--resume)
"$python_bin" -u research/a6/run.py --config "$group_dir/A6-config.json" --id "$run_id" --screen --screen-updates 50000 "${extra[@]}"
printf 'COMPARING\n' > "$group_dir/status.txt"
"$python_bin" research/a6/compare.py --audit "$group_dir/reuse-audit.json" --candidate-run "$PWD/runs/$run_id" --output "$group_dir/comparison-50000.json"
printf 'BACKING_UP\n' > "$group_dir/status.txt"
"$python_bin" research/a6/backup.py --project "$PWD" --group-dir "$group_dir" --run-id "$run_id" \
    --destination "${BACKUP_ROOT:-/home/featurize/work/span-x4-research/records}" --receipt "$group_dir/backup-path.txt"
printf 'COMPLETE_A6; discuss PSNR and attention sharing; instance retained\n' > "$group_dir/status.txt"
printf '\nREMINDER: A6 finished. Review PSNR versus B0 first, then discuss whether to share attention.\n'
