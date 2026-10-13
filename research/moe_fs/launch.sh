#!/usr/bin/env bash
# bash research/moe_fs/launch.sh GROUP DATA_ROOT [UPDATES] [MICRO] [BACKEND] [EVAL_EVERY]
set -euo pipefail
cd "$(dirname "$0")/../.."
group=${1:?group required}
data_root=${2:?data root required}
updates=${3:-5000}
micro=${4:-16}
backend=${5:-npy_mmap}
eval_every=${6:-1000}
python_bin=${PYTHON_BIN:-python}
auto_release=${AUTO_RELEASE:-0}
backup_root=${BACKUP_ROOT:-/home/featurize/work/span-x4-research/records}
[[ "$auto_release" == 0 || "$auto_release" == 1 ]] || { printf 'AUTO_RELEASE must be 0 or 1.\n'; exit 2; }
if [[ "$auto_release" == 1 ]]; then
    command -v featurize >/dev/null
    [[ "$backup_root" == /home/featurize/work/* ]] || { printf 'Backup must be on persistent work storage.\n'; exit 2; }
    mkdir -p "$backup_root"
    [[ -w "$backup_root" ]] || exit 2
    "$python_bin" -c 'import matplotlib; print("Plot dependency ready:", matplotlib.__version__)'
fi
[[ $(uname -s) == Linux ]] || { printf 'Linux server required.\n'; exit 2; }
command -v flock >/dev/null
command -v nohup >/dev/null
args=(--group "$group" --data-root "$data_root" --updates "$updates" --micro-batch "$micro" --image-backend "$backend" --eval-every "$eval_every")
"$python_bin" research/moe_fs/screen.py "${args[@]}" --prepare-only
job="logs/moe_fs/$group/job-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$job"
PYTHON_BIN="$python_bin" AUTO_RELEASE="$auto_release" BACKUP_ROOT="$backup_root" MOE_FS_GROUP="$group" nohup bash research/moe_fs/worker.sh "$job" "${args[@]}" > "$job/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
printf 'PID=%s\nLog=%s/nohup.log\nStatus=%s/status.txt\n' "$pid" "$job" "$job"
