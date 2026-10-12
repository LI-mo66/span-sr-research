#!/usr/bin/env bash
set -euo pipefail
group=$1
job="logs/server_jobs/$group"
exec 9>logs/server-gpu.lock
if ! flock -n 9; then
  echo 'GPU lock held; exit without shutdown'
  exit 75
fi
finish() {
  code=$?
  trap - EXIT
  printf 'exit_code=%s\nfinished_utc=%s\n' "$code" "$(date -u +%FT%TZ)" > "$job/status.txt"
  sync
  if ! /usr/bin/shutdown; then
    echo 'SHUTDOWN_FAILED: power off through platform console' | tee -a "$job/status.txt"
  fi
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUBLAS_WORKSPACE_CONFIG=:4096:8
id="$group-B0-s10"
resume=()
if [ -f "runs/$id/latest.pt" ]; then resume=(--resume); fi
python -u research/train.py --config "$job/B0-s10.local.json" --id "$id" --screen --screen-updates 20000 "${resume[@]}"
