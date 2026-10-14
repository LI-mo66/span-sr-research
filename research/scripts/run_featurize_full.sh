#!/usr/bin/env bash
set -euo pipefail
id=$1
job="logs/server_jobs/$id"
exec 9>logs/server-gpu.lock
if ! flock -n 9; then
  echo 'GPU lock held; refusing duplicate task' > "$job/status-duplicate-$(date -u +%Y%m%dT%H%M%SZ).txt"
  exit 75
fi
finish() {
  code=$?
  trap - EXIT
  printf 'exit_code=%s\nfinished_utc=%s\n' "$code" "$(date -u +%FT%TZ)" > "$job/status.txt"
  sync
  echo "Training exited: $code. No provider shutdown hook configured."
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUBLAS_WORKSPACE_CONFIG=:4096:8
resume=()
if [ -f "runs/$id/latest.pt" ]; then resume=(--resume); fi
python -u research/train.py --config "$job/baseline.local.json" --id "$id" "${resume[@]}"
