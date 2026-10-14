#!/usr/bin/env bash
# EASA-only 50k screen; archived B0 is read-only comparison evidence.
set -euo pipefail
cd "$(dirname "$0")/../.."
group=${1:?Usage: bash research/easa/launch_reuse.sh GROUP BASELINE_RUN}
baseline=${2:?Archived B0 run required}
[[ "$group" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
[[ $(uname -s) == Linux ]] || { printf 'Linux server required.\n'; exit 2; }
auto_release=${AUTO_RELEASE:-0}
backup_root=${BACKUP_ROOT:-/home/featurize/work/span-x4-research/records}
[[ "$auto_release" == 0 || "$auto_release" == 1 ]] || exit 2
if [[ "$auto_release" == 1 ]]; then
    command -v featurize >/dev/null
    "$python_bin" -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); assert p.is_relative_to(pathlib.Path("/home/featurize/work")), "Backup must be on persistent work storage"' "$backup_root"
    mkdir -p "$backup_root"
    [[ -w "$backup_root" ]] || exit 2
fi
group_dir="$PWD/logs/easa/$group"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_bin" research/easa/prepare_reuse.py --baseline-run "$baseline" --output-dir "$group_dir"
# Recheck the exact server CUDA/PyTorch version; no real baseline training.
"$python_bin" research/easa/verify.py --device cuda --output "$group_dir/engineering-cuda.json" > "$group_dir/engineering-cuda.log" 2>&1
run_id="$group-E1-mid-s10"
"$python_bin" research/easa/run.py --config "$group_dir/E1-config.json" --id "$run_id" --preflight > "$group_dir/preflight.log" 2>&1
PYTHON_BIN="$python_bin" AUTO_RELEASE="$auto_release" BACKUP_ROOT="$backup_root" nohup bash research/easa/worker_reuse.sh "$group_dir" "$run_id" > "$group_dir/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$group_dir/pid"
printf 'EASA only. PID=%s\nLog=%s/nohup.log\nStatus=%s/status.txt\n' "$pid" "$group_dir" "$group_dir"
