#!/usr/bin/env bash
# bash research/moe_fs/launch.sh GROUP DATA_ROOT [UPDATES] [MICRO] [BACKEND]
set -euo pipefail
cd "$(dirname "$0")/../.."
group=${1:?group required}
data_root=${2:?data root required}
updates=${3:-5000}
micro=${4:-16}
backend=${5:-npy_mmap}
python_bin=${PYTHON_BIN:-python}
[[ $(uname -s) == Linux ]] || { printf 'Linux server required.\n'; exit 2; }
command -v flock >/dev/null
command -v nohup >/dev/null
args=(--group "$group" --data-root "$data_root" --updates "$updates" --micro-batch "$micro" --image-backend "$backend")
"$python_bin" research/moe_fs/screen.py "${args[@]}" --prepare-only
job="logs/moe_fs/$group/job-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$job"
PYTHON_BIN="$python_bin" nohup bash research/moe_fs/worker.sh "$job" "${args[@]}" > "$job/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
printf 'PID=%s\nLog=%s/nohup.log\nStatus=%s/status.txt\n' "$pid" "$job" "$job"
