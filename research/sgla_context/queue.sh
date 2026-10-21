#!/usr/bin/env bash
# Three independent scratch runs; release only after all verified backups.
set -Eeuo pipefail
group=${1:?Group required}
reference=${2:?SG6 reference archive required}
[[ "$group" =~ ^[A-Za-z0-9_-]+$ ]]
cd "$(dirname "$0")/../.."
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
backup_root=${BACKUP_ROOT:?Persistent backup root required}
queue_dir="logs/sgla_context/$group-queue"
[[ ! -e "$queue_dir" ]] || { echo 'Queue already exists; preserving it.'; exit 1; }
mkdir -p "$queue_dir"
exec 8>logs/sgla_context/all-queue.lock
flock -n 8 || { echo 'Another SGLA queue is active.'; exit 1; }
command -v featurize >/dev/null
trap 'rc=$?; printf "FAILED exit=%s line=%s; instance retained\n" "$rc" "$LINENO" | tee "$queue_dir/status.txt"; exit "$rc"' ERR
printf 'RUNNING context original local; scratch seed10 50K each\n' > "$queue_dir/status.txt"
for mode in context original local; do
  printf 'START %s\n' "$mode" | tee -a "$queue_dir/progress.txt"
  bash research/sgla_context/launch.sh "$group" "$reference" "$mode"
  config_dir="logs/sgla_context/$group-$mode"
  backup=$(cat "$config_dir/backup-path.txt")
  [[ -f "$backup/BACKUP_COMPLETE" ]]
  [[ $(cat "$config_dir/training-exit-code.txt") == 0 ]]
  [[ $(cat "$config_dir/comparison-exit-code.txt") == 0 ]]
  printf '%s\t%s\n' "$mode" "$backup" >> "$queue_dir/archives.tsv"
  printf 'COMPLETE %s\n' "$mode" | tee -a "$queue_dir/progress.txt"
done
"$python_bin" research/sgla_context/summarize_queue.py --group "$group" --output "$queue_dir/summary.json"
printf 'COMPLETE; ARCHIVING_QUEUE\n' > "$queue_dir/status.txt"
queue_backup="$backup_root/$group-queue-$(date -u +%Y%m%dT%H%M%SZ)"
# Queue has no own training run; existing backup helper copies queue logs+source.
"$python_bin" research/sgla_context/backup.py --run "runs/$group-no-queue-training" --logs "$queue_dir" --destination "$queue_backup" --training-exit 0 --comparison-exit 0
[[ -f "$queue_backup/BACKUP_COMPLETE" ]]
printf 'COMPLETE; ALL_BACKUPS_VERIFIED\n' | tee "$queue_dir/status.txt" "$queue_backup/release-status.txt"
sync
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$busy" ]]; then
  printf 'RELEASE_SKIPPED_GPU_BUSY: %s\n' "$busy" | tee -a "$queue_dir/status.txt" "$queue_backup/release-status.txt"
  exit 0
fi
printf 'RELEASE_REQUESTED\n' | tee -a "$queue_dir/status.txt" "$queue_backup/release-status.txt"
sync
if featurize instance release < /dev/null; then
  release_rc=0
else
  release_rc=$?
fi
printf 'release_command_exit_code=%s\n' "$release_rc" >> "$queue_backup/release-status.txt"
if (( release_rc != 0 )); then
  printf 'RELEASE_FAILED\n' | tee -a "$queue_dir/status.txt" "$queue_backup/release-status.txt"
  exit 73
fi
