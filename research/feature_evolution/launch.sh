#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]] || { echo 'Linux GPU server required'; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
DATA_ROOT=${DATA_ROOT:-/home/featurize/datasets}
REFERENCE_BACKUP=${REFERENCE_BACKUP:-/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/featurize/work/sg6-fe-records}
CACHE_ROOT=${CACHE_ROOT:-$DATA_ROOT/image_cache}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUNTIME_ROOT=${RUNTIME_ROOT:-/home/featurize/span-sg6-fe-runtime/$STAMP}
GROUP=$OUTPUT_ROOT/SG6-FE-$STAMP
for tool in git nohup nvidia-smi; do command -v "$tool" >/dev/null; done
[[ -x "$PYTHON_BIN" && -d "$DATA_ROOT" && -f "$REFERENCE_BACKUP/BACKUP_COMPLETE" ]] || {
  echo 'Python/data/fixed SG6 reference missing; training not submitted'; exit 2;
}
[[ -z $(git status --porcelain) ]] || { git status --short; exit 2; }
[[ ! -e "$RUNTIME_ROOT" && ! -e "$GROUP" ]] || { echo 'Existing output preserved'; exit 2; }
for target in runs logs datasets; do
  [[ ! -e "$target" && ! -L "$target" ]] || { echo "Use a fresh checkout; existing $target preserved"; exit 2; }
done
mkdir -p "$RUNTIME_ROOT/runs" "$RUNTIME_ROOT/logs" "$CACHE_ROOT" "$GROUP" datasets
ln -s "$RUNTIME_ROOT/runs" runs
ln -s "$RUNTIME_ROOT/logs" logs
ln -s "$CACHE_ROOT" datasets/image_cache
printf '\n/runs\n/logs\n/datasets\n/tmp\n' >> .git/info/exclude
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export FE_OBSERVE=1
nohup "$PYTHON_BIN" -B -u research/feature_evolution/server.py \
  --group "$GROUP" --reference-backup "$REFERENCE_BACKUP" --data-root "$DATA_ROOT" --shutdown \
  > "$GROUP/nohup.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$GROUP/PID"
sleep 2
kill -0 "$PID" 2>/dev/null || { tail -n 80 "$GROUP/nohup.log"; exit 1; }
printf 'PID=%s\nSOURCE=%s\nGROUP=%s\nLOG=%s/nohup.log\n' "$PID" "$PWD" "$GROUP" "$GROUP"
printf 'Serial queue: add 50K -> difference 50K, scratch seed10; fixed SG6 reused after audit.\n'
printf 'View: tail -f "%s/nohup.log"\n' "$GROUP"
printf 'Return ZIP: %s/SG6-FE-RETURN.zip\n' "$GROUP"
printf 'Full weights: %s/{add,difference}-backup\n' "$GROUP"
printf 'AUTO SHUTDOWN ENABLED: only after both runs, comparisons and verified backups succeed.\n'
printf 'Any failure retains the server and logs. This uses system shutdown, not provider release.\n'
