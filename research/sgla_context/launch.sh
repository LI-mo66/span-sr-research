#!/usr/bin/env bash
# Foreground one-candidate 50K run. No shutdown, release, baseline or extra arms.
set -euo pipefail
group=${1:?usage: launch.sh GROUP SG6_ARCHIVE [context|original|local]}
reference=${2:?SG6 archive required}
mode=${3:-context}
[[ "$group" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$mode" == context || "$mode" == original || "$mode" == local ]]
cd "$(dirname "$0")/../.."
python_bin=${PYTHON_BIN:-python3}
config_dir="logs/sgla_context/$group-$mode"
run_id="$group-SGLA-$mode-s10"
[[ ! -e "runs/$run_id" && ! -e "$config_dir" ]] || { echo 'Output exists; preserving it.'; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo 'Commit source before launching.'; exit 1; }
command -v flock >/dev/null
mkdir -p logs/sgla_context
exec 9>logs/sgla_context/queue.lock
flock -n 9 || { echo 'SGLA queue already active.'; exit 1; }
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > logs/sgla_context/gpu-before.txt
[[ ! -s logs/sgla_context/gpu-before.txt ]] || { echo 'GPU compute processes present; no training started.'; cat logs/sgla_context/gpu-before.txt; exit 1; }
"$python_bin" research/sgla_context/verify.py --device cuda --output "logs/sgla_context/$group-$mode-engineering.json"
"$python_bin" research/sgla_context/resources.py --output "logs/sgla_context/$group-$mode-resources.json"
"$python_bin" research/sgla_context/prepare.py --reference-backup "$reference" --output "$config_dir" --mode "$mode"
set +e
"$python_bin" -u research/sgla_context/run.py --config "$config_dir/config.json" --id "$run_id" --screen --screen-updates 50000
training_exit=$?
set -e
printf '%s\n' "$training_exit" > "$config_dir/training-exit-code.txt"
comparison_exit=1
if [[ "$training_exit" == 0 ]]; then
  set +e
  "$python_bin" research/sgla_context/compare.py --audit "$config_dir/audit.json" --run "runs/$run_id" --output "$config_dir/comparison-50000.json"
  comparison_exit=$?
  set -e
fi
printf '%s\n' "$comparison_exit" > "$config_dir/comparison-exit-code.txt"
backup_root=${BACKUP_ROOT:-$(dirname "$reference")}
destination="$backup_root/$run_id-$(date -u +%Y%m%dT%H%M%SZ)"
"$python_bin" research/sgla_context/backup.py --run "runs/$run_id" --logs "$config_dir" --destination "$destination" --training-exit "$training_exit" --comparison-exit "$comparison_exit"
[[ "$training_exit" == 0 && "$comparison_exit" == 0 ]] || { echo 'Run failed; failure archive retained.'; exit 1; }
echo "Completed and hash-verified backup: $destination. Instance remains running."
