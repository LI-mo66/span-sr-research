#!/usr/bin/env bash
# Only A6 is trained. Archived B0 is read-only comparison evidence.
set -euo pipefail
cd "$(dirname "$0")/../.."
group=${1:?Usage: bash research/a6/launch.sh GROUP ARCHIVED_B0_RUN}
baseline=${2:?Archived B0 run required}
[[ "$group" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ $(uname -s) == Linux ]] || { printf 'Linux server required.\n'; exit 2; }
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
group_dir="$PWD/logs/a6/$group"
[[ ! -e "$group_dir" && ! -e "$PWD/runs/$group-A6-s10" ]] || { printf 'Group/run exists; use resume.sh or a new ID.\n'; exit 2; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { printf 'GPU already has compute processes:\n%s\n' "$busy"; exit 2; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_bin" research/a6/prepare_reuse.py --baseline-run "$baseline" --output-dir "$group_dir"
printf 'ENGINEERING_CHECK\n' > "$group_dir/status.txt"
"$python_bin" research/a6/verify.py --device cuda --output "$group_dir/engineering-cuda.json" > "$group_dir/engineering-cuda.log" 2>&1
"$python_bin" research/a6/run.py --config "$group_dir/A6-config.json" --id "$group-A6-s10" --preflight > "$group_dir/preflight.log" 2>&1
# Include the separate b546036 evidence history as well as candidate history.
git bundle create "$group_dir/source.bundle" --all
PYTHON_BIN="$python_bin" nohup bash research/a6/worker.sh "$group_dir" "$group-A6-s10" > "$group_dir/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$group_dir/pid"
printf 'A6 ONLY, from scratch, 50k updates; no B0 training.\nPID=%s\nLog=%s/nohup.log\n' "$pid" "$group_dir"
