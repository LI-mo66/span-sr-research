#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/environment/miniconda3/bin/python3}"
CHECKOUT_DIR="${CHECKOUT_DIR:-/home/featurize/work/span-sg6-strip-v1}"
REFERENCE_BACKUP="${REFERENCE_BACKUP:-/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269}"
DATA_ROOT="${DATA_ROOT:-/home/featurize/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/featurize/work/sg6-strip-records}"
CACHE_ROOT="${CACHE_ROOT:-$DATA_ROOT/image_cache}"
MAX_TIME_RATIO="${MAX_TIME_RATIO:-1.10}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN"; exit 1; }
[[ ! -e "$CHECKOUT_DIR" ]] || { echo "Refuse overwrite: $CHECKOUT_DIR"; exit 1; }
[[ -d "$REFERENCE_BACKUP" ]] || { echo "Missing SG6 reference: $REFERENCE_BACKUP"; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "Missing data: $DATA_ROOT"; exit 1; }
cd "$HERE"
sha256sum -c SHA256SUMS
git bundle list-heads sg6-strip.bundle
git clone --branch research/sg6-partial-strip-v1 "$HERE/sg6-strip.bundle" "$CHECKOUT_DIR"
git -C "$CHECKOUT_DIR" bundle verify "$HERE/sg6-strip.bundle"
git -C "$CHECKOUT_DIR" remote set-url origin https://github.com/LI-mo66/span-sr-research.git
mkdir -p "$CHECKOUT_DIR/datasets" "$CHECKOUT_DIR/logs" "$CACHE_ROOT" "$OUTPUT_ROOT"
ln -s "$CACHE_ROOT" "$CHECKOUT_DIR/datasets/image_cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG="$CHECKOUT_DIR/logs/queue-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup "$PYTHON_BIN" -B -u "$CHECKOUT_DIR/research/sg6_strip/server.py" \
  --reference-backup "$REFERENCE_BACKUP" --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" --max-time-ratio "$MAX_TIME_RATIO" \
  >"$LOG" 2>&1 < /dev/null &
PID=$!
printf 'Queue PID: %s\nLog: %s\nInspect: tail -f "%s"\n' "$PID" "$LOG" "$LOG"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then tail -n 60 "$LOG"; exit 1; fi
