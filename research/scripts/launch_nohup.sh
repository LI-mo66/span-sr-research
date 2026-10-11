#!/usr/bin/env bash
# Usage: bash research/scripts/launch_nohup.sh GROUP DATA_ROOT initial|confirm|fusion [MICROBATCH] [png|npy_mmap]
set -euo pipefail
if [[ $(uname -s) != Linux ]]; then printf 'Run this launcher on the Linux rental server.\n'; exit 2; fi
cd "$(dirname "$0")/../.."
group=${1:?group ID required}
data_root=${2:?dataset root required}
phase=${3:-initial}
micro=${4:-16}
backend=${5:-npy_mmap}
python_bin=${PYTHON_BIN:-python}
shutdown_bin=${SHUTDOWN_BIN:-/usr/bin/shutdown}
command -v nohup >/dev/null
command -v flock >/dev/null
[[ -x "$shutdown_bin" ]] || { printf 'Shutdown executable unavailable: %s\n' "$shutdown_bin"; exit 2; }
[[ -z $(git status --porcelain) ]] || { printf 'Commit/stash source edits first.\n'; exit 2; }
"$python_bin" -c 'import torch,cv2,numpy; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__,torch.cuda.get_device_name())'
args=(--group "$group" --data-root "$data_root" --phase "$phase" --micro-batch "$micro" --image-backend "$backend")
"$python_bin" research/server_queue.py "${args[@]}" --prepare-only
job="logs/server_jobs/$group/${phase}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$job"
PYTHON_BIN="$python_bin" SHUTDOWN_BIN="$shutdown_bin" nohup bash research/scripts/run_server_worker.sh "$job" "${args[@]}" > "$job/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
printf 'Started PID=%s\nLog=%s/nohup.log\nThis batch invokes shutdown on success OR failure.\n' "$pid" "$job"
