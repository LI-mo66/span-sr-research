#!/usr/bin/env bash
set -euo pipefail
base=/home/featurize/work/span-x4-research
repo="$base/code/span-sr-research"
code="$base/code/span-paper-visuals-sci-final-20261128"
branch=research/paper-visuals-sci-final-20261128
python_bin=/environment/miniconda3/bin/python3

export PAPER_BASE="$base"
export PAPER_PROJECT="$base/code/span-sg6-f0-long-x8-1m"
export PAPER_ROOT="$base/records/paper_visuals_20261126"
export PAPER_OLD_OUT="$PAPER_ROOT/step10-paper-suite-v2-01"
export PAPER_OUT="$PAPER_ROOT/step11-paper-sci-final-01"
export PAPER_DATA=/home/featurize/datasets/benchmark
export PYTHON_BIN="$python_bin"
export PYTHONDONTWRITEBYTECODE=1

test -x "$python_bin"
test -d "$repo"

# Quick pre-check: 5 benchmarks + CUDA GPU
"$python_bin" - <<'PY'
import os, torch
from pathlib import Path
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA GPU unavailable! Please run on RTX 4090 instance.')
print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__)
data = Path(os.environ['PAPER_DATA'])
missing = []
for name, count in [('Set5', 5), ('Set14', 14), ('B100', 100), ('Urban100', 100), ('Manga109', 109)]:
    opts = [p for p in data.iterdir() if p.is_dir() and p.name.casefold() == name.casefold()] if data.is_dir() else []
    if len(opts) != 1:
        missing.append(str(data / name)); continue
    for rel in ['HR', 'LR_bicubic/X4']:
        p = opts[0] / rel
        if len(list(p.glob('*.png'))) != count:
            missing.append(f'{p} (expected {count})')
if missing:
    print('MISSING DATASETS:\n' + '\n'.join(missing))
    raise SystemExit(1)
print('PASS: 5 benchmark datasets verified.')
PY

for attempt in 1 2 3; do
    if git -C "$repo" fetch origin "$branch"; then break; fi
    if [[ "$attempt" == 3 ]]; then exit 1; fi
    sleep 3
done
target_commit=$(git -C "$repo" rev-parse "origin/$branch")
if [[ -e "$code" ]]; then
    git -C "$code" checkout --detach "$target_commit"
else
    git -C "$repo" worktree add --detach "$code" "$target_commit"
fi
bash -n "$code/research/paper_visuals_sci_final/launch_sci.sh"

mkdir -p "$PAPER_OUT"
if [[ -f "$PAPER_OUT/COMPUTE_AND_SERVER_BACKUP_COMPLETE" ]]; then
    echo "本轮已全部完成：$PAPER_OUT"
    exit 0
fi
if ! flock -n "$PAPER_OUT/suite.lock" true; then
    echo "队列已在运行中，未重复启动。查看进度：tail -f $PAPER_OUT/queue.nohup.log"
    exit 0
fi

nohup bash "$code/research/paper_visuals_sci_final/launch_sci.sh" \
    >> "$PAPER_OUT/queue.nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PAPER_OUT/suite.pid"
printf '已后台启动 PID=%s (commit=%s)\n' "$pid" "$target_commit"
printf '输出目录：%s\n' "$PAPER_OUT"
printf '实时监控：tail -f %s/queue.nohup.log\n' "$PAPER_OUT"
printf '审核压缩包（完成后生成）：%s/paper-visuals-sci-final-review.zip\n' "$PAPER_OUT"
