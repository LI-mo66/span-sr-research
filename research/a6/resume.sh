#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
group=${1:?Usage: bash research/a6/resume.sh GROUP}
[[ "$group" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
group_dir="$PWD/logs/a6/$group"
[[ -f "$group_dir/reuse-audit.json" && -f "$PWD/runs/$group-A6-s10/latest.pt" ]] || exit 2
if [[ -f "$group_dir/pid" ]] && kill -0 "$(cat "$group_dir/pid")" 2>/dev/null; then
    printf 'Recorded worker PID still exists; inspect before resuming.\n'; exit 2
fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { printf 'GPU busy; resume refused.\n'; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-/environment/miniconda3/bin/python3}" nohup bash research/a6/worker.sh "$group_dir" "$group-A6-s10" resume >> "$group_dir/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$group_dir/pid"
printf 'Resuming A6 from its own latest.pt to cumulative 50k.\n'
