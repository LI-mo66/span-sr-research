#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
id=${1:-SPAN52-X4-FULL-S10}
data=${2:-/home/featurize/datasets}
command -v nohup >/dev/null
command -v flock >/dev/null
test -z "$(git status --porcelain)" || { echo 'Source is dirty; stop'; exit 2; }
python -c 'import torch,cv2,numpy; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'
python research/prepare_featurize_baseline.py --id "$id" --data-root "$data"
job="logs/server_jobs/$id"
# This full data audit happens before background launch; it can take several minutes.
python research/train.py --config "$job/baseline.local.json" --id "$id" --preflight
if [ -d "runs/$id" ] && [ ! -f "runs/$id/latest.pt" ]; then
  echo 'Run exists without a checkpoint; inspect it or use a fresh ID'; exit 2
fi
log="$job/nohup-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash research/scripts/run_featurize_full.sh "$id" > "$log" 2>&1 < /dev/null &
echo "$!" > "$job/pid"
printf 'PID=%s\nLog=%s/%s\nFull 1M updates; no platform shutdown hook configured.\n' "$(cat "$job/pid")" "$PWD" "$log"
