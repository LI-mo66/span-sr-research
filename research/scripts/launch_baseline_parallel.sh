#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
group=${1:-SPAN52-BASELINE-02}
data=${2:-$PWD/datasets}
python research/server_queue.py --group "$group" --data-root "$data" --micro-batch 16 --image-backend png --prepare-only
python - "$group" <<'PY'
import json,sys
from pathlib import Path
root=Path.cwd(); group=root/'logs/server_jobs'/sys.argv[1]
cfg=json.loads((root/'research/configs/baseline_x4.json').read_text())
cfg.update(seed=10,micro_batch=16,image_backend='png',variants=[],loader_workers=8,
           loader_prefetch=2,cpu_threads=4,log_updates=10,
           loader_sampling='indexed-v1',paths_file=str(group/'paths.local.json'))
path=group/'B0-s10.local.json'
if path.exists():
    assert json.loads(path.read_text())==cfg, 'Frozen config differs'
else: path.write_text(json.dumps(cfg,indent=2))
PY
test -x /usr/bin/shutdown
command -v flock >/dev/null
test -z "$(git status --porcelain)"
job="logs/server_jobs/$group"
log="$job/nohup-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash research/scripts/run_baseline_worker.sh "$group" > "$log" 2>&1 < /dev/null &
echo "$!" > "$job/pid"
printf 'PID=%s\nLog=%s/%s\n' "$(cat "$job/pid")" "$PWD" "$log"
