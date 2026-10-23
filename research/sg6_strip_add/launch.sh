#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]] || { echo 'Linux server required'; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-/environment/miniconda3/bin/python3}"
REFERENCE_BACKUP="${REFERENCE_BACKUP:-/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269}"
K31_REFERENCE="${K31_REFERENCE:-/home/featurize/work/span-sg6-strip-v1.6rwSo3/runs/SG6-STRIP-20261022T075259Z-K31-B3-s10}"
DATA_ROOT="${DATA_ROOT:-/home/featurize/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/featurize/work/sg6-stripadd-records}"
CACHE_ROOT="${CACHE_ROOT:-$DATA_ROOT/image_cache}"
[[ -x "$PYTHON_BIN" && -d "$DATA_ROOT" && -f "$REFERENCE_BACKUP/BACKUP_COMPLETE" && -f "$K31_REFERENCE/manifest.json" ]] || { echo 'Python/data/SG6/K31 reference missing'; exit 2; }
[[ -z $(git status --porcelain) ]] || { echo 'Source must be clean'; exit 2; }
mkdir -p datasets logs "$CACHE_ROOT" "$OUTPUT_ROOT"
if [[ -L datasets/image_cache ]]; then
  [[ $(readlink -f datasets/image_cache) == $(readlink -f "$CACHE_ROOT") ]] || { echo 'Cache link differs'; exit 2; }
elif [[ -e datasets/image_cache ]]; then
  echo 'Existing cache preserved; use fresh checkout'; exit 2
else
  ln -s "$CACHE_ROOT" datasets/image_cache
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG="$(pwd)/logs/queue-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
nohup "$PYTHON_BIN" -B -u research/sg6_strip_add/server.py \
  --reference-backup "$REFERENCE_BACKUP" --k31-reference "$K31_REFERENCE" \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
  --max-time-ratio "${MAX_TIME_RATIO:-1.10}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf 'PID=%s\nLOG=%s\nAutomatic shutdown: DISABLED\nInspect: tail -n 80 -f "%s"\n' "$PID" "$LOG" "$LOG"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then tail -n 80 "$LOG"; exit 1; fi
