#!/usr/bin/env bash
# One isolated SG6+ATD run. Existing SG6 archive is read-only.
set -euo pipefail
cd "$(dirname "$0")/../.."
group=${1:?GROUP required}
reference=${2:?SG6 archive required}
[[ "$group" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
backup_root=${BACKUP_ROOT:-/home/featurize/work/span-x4-research/records}
auto_release=${AUTO_RELEASE:-1}
[[ $(uname -s) == Linux ]] || { echo 'Linux server required'; exit 2; }
[[ "$auto_release" == 0 || "$auto_release" == 1 ]] || exit 2
[[ -z $(git status --porcelain) ]] || { echo 'Source must be clean'; exit 2; }
run_id="$group-A1-tail-s10"
group_dir="$PWD/logs/sg6_atd/$group"
[[ ! -e "$group_dir" && ! -e "runs/$run_id" ]] || { echo 'Existing experiment preserved; refusing duplicate'; exit 2; }
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo 'Another research GPU job holds the lock'; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { printf 'GPU compute processes already active; not starting: %s\n' "$busy"; exit 75; }
if [[ "$auto_release" == 1 ]]; then
    command -v featurize >/dev/null
    "$python_bin" -c 'from pathlib import Path; import sys; assert Path(sys.argv[1]).resolve().is_relative_to(Path("/home/featurize/work"))' "$backup_root"
fi
mkdir -p "$backup_root"
[[ -w "$backup_root" ]] || exit 2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python_bin" -u research/sg6_atd/prepare_reuse.py --reference-backup "$reference" --output "$group_dir"
trap 'rc=$?; if (( rc != 0 )); then printf "PREFLIGHT_FAILED exit=%s\n" "$rc" > "$group_dir/status.txt"; fi' EXIT
printf 'PREFLIGHT\n' > "$group_dir/status.txt"
"$python_bin" -u research/sg6_atd/check_reference.py --device cuda --output "$group_dir/reference-cuda.json" > "$group_dir/reference-cuda.log" 2>&1
"$python_bin" -u research/sg6_atd/verify.py --device cuda --config "$group_dir/A1-config.json" --output "$group_dir/engineering-cuda.json" > "$group_dir/engineering-cuda.log" 2>&1
"$python_bin" -u research/sg6_atd/server_smoke.py --config "$group_dir/A1-config.json" --output "$group_dir/server-smoke.json" > "$group_dir/server-smoke.log" 2>&1
"$python_bin" -u research/sg6_atd/resources.py --config "$group_dir/A1-config.json" --output "$group_dir/resources.json" > "$group_dir/resources.log" 2>&1
"$python_bin" -u research/sg6_atd/run.py --config "$group_dir/A1-config.json" --id "$run_id" --preflight > "$group_dir/preflight.log" 2>&1
PYTHON_BIN="$python_bin" AUTO_RELEASE="$auto_release" BACKUP_ROOT="$backup_root" nohup bash research/sg6_atd/worker_reuse.sh "$group_dir" "$run_id" > "$group_dir/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$group_dir/pid"
printf 'SG6+ATD only. PID=%s\nLog=%s/nohup.log\n' "$pid" "$group_dir"
