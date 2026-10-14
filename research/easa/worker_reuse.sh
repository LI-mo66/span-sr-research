#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
group_dir=${1:?Group directory required}
run_id=${2:?Run ID required}
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'RUNNING\n' > "$group_dir/status.txt"
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED exit=%s\n" "$rc" > "$group_dir/status.txt"; fi' EXIT
"$python_bin" -u research/easa/run.py --config "$group_dir/E1-config.json" --id "$run_id" --screen --screen-updates 50000
"$python_bin" research/easa/compare_reuse.py --audit "$group_dir/reuse-audit.json" --candidate-run "$PWD/runs/$run_id" --output "$group_dir/comparison-50000.json"
printf 'COMPLETE\n' > "$group_dir/status.txt"
