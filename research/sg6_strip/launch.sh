#!/usr/bin/env bash
# Run from a fresh, pinned checkout. No cloning/updating of an active checkout.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]] || { echo 'Linux server required'; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-/environment/miniconda3/bin/python3}"
REFERENCE_BACKUP="${REFERENCE_BACKUP:-/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269}"
DATA_ROOT="${DATA_ROOT:-/home/featurize/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/featurize/work/sg6-strip-records}"
CACHE_ROOT="${CACHE_ROOT:-$DATA_ROOT/image_cache}"
MAX_TIME_RATIO="${MAX_TIME_RATIO:-1.10}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
[[ -x "$PYTHON_BIN" && -f "$REFERENCE_BACKUP/BACKUP_COMPLETE" && -d "$DATA_ROOT" ]] || { echo 'Python/reference/data missing'; exit 2; }
[[ -z $(git status --porcelain) ]] || { echo 'Source must be clean'; exit 2; }
mkdir -p datasets logs "$CACHE_ROOT" "$OUTPUT_ROOT"
if [[ -L datasets/image_cache ]]; then
  [[ $(readlink -f datasets/image_cache) == $(readlink -f "$CACHE_ROOT") ]] || { echo 'Existing cache link differs; preserved'; exit 2; }
elif [[ -e datasets/image_cache ]]; then
  echo 'Existing real cache directory preserved; use a fresh checkout'; exit 2
else
  ln -s "$CACHE_ROOT" datasets/image_cache
fi
args=(--reference-backup "$REFERENCE_BACKUP" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --max-time-ratio "$MAX_TIME_RATIO")
case "$AUTO_SHUTDOWN" in
  1) args+=(--shutdown-after-queue); [[ -z ${SHUTDOWN_BIN:-} ]] || args+=(--shutdown-bin "$SHUTDOWN_BIN");;
  0) ;;
  *) echo 'AUTO_SHUTDOWN must be 0 or 1'; exit 2;;
esac
LOG="$(pwd)/logs/queue-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nohup "$PYTHON_BIN" -B -u research/sg6_strip/server.py "${args[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf 'PID=%s\nLOG=%s\nAUTO_SHUTDOWN=%s\nInspect: tail -f "%s"\n' "$PID" "$LOG" "$AUTO_SHUTDOWN" "$LOG"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then tail -n 80 "$LOG"; exit 1; fi
