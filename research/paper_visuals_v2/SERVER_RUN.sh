#!/usr/bin/env bash
set -euo pipefail
base=/home/featurize/work/span-x4-research
repo="$base/code/span-sr-research"
code="$base/code/span-paper-visuals-v2-20261127"
branch=research/paper-visuals-suite-v2-20261127
commit=1441a19596d7b969ca1a6675b8a732a70d65e6f4
python_bin=/environment/miniconda3/bin/python3

export PAPER_BASE="$base"
export PAPER_PROJECT="$base/code/span-sg6-f0-long-x8-1m"
export PAPER_ROOT="$base/records/paper_visuals_20261126"
export PAPER_OUT="$PAPER_ROOT/step10-paper-suite-v2-01"
export PAPER_DATA=/home/featurize/datasets/benchmark
export PYTHON_BIN="$python_bin"
export PYTHONDONTWRITEBYTECODE=1

test -x "$python_bin"
test -d "$repo"

# Fail before inference if the previous instance lost nonpersistent test data.
"$python_bin" - <<'PY'
import os
from pathlib import Path
root=Path(os.environ['PAPER_ROOT']);data=Path(os.environ['PAPER_DATA'])
missing=[]
for p in [root/'step8-roi-review/roi_ranking.json',Path(os.environ['PAPER_PROJECT'])/'research/runtime.py']:
    if not p.is_file():missing.append(str(p))
for name,count in [('Set14',14),('Urban100',100),('Manga109',109),('Set5',5)]:
    options=[p for p in data.iterdir() if p.name.casefold()==name.casefold() and p.is_dir()] if data.is_dir() else []
    if len(options)!=1:
        missing.append(str(data/name));continue
    for rel in ['HR','LR_bicubic/X4']:
        p=options[0]/rel
        if len(list(p.glob('*.png')))!=count:missing.append(str(p)+' expected '+str(count)+' PNG')
if missing:
    print('缺少原任务文件或测试集，尚未开始推理，请先恢复以下路径：')
    print('\n'.join(missing));raise SystemExit(1)
print('PASS: 原任务入口和四集合图像数量；内容哈希稍后由后台核对。')
PY

for attempt in 1 2 3; do
    if git -C "$repo" fetch origin "$branch"; then break; fi
    if [[ "$attempt" == 3 ]]; then exit 1; fi
    sleep 3
done
if [[ ! -e "$code" ]]; then
    git -C "$repo" worktree add --detach "$code" "$commit"
fi
[[ $(git -C "$code" rev-parse HEAD) == "$commit" ]]
[[ -z $(git -C "$code" diff --name-only HEAD -- research/paper_visuals_v2) ]]
bash -n "$code/research/paper_visuals_v2/launch.sh"
mkdir -p "$PAPER_OUT"
if [[ -f "$PAPER_OUT/COMPUTE_AND_SERVER_BACKUP_COMPLETE" ]]; then
    echo '本轮已完成，无需重复提交。请查看状态及备份路径。'
    exit 0
fi
if ! flock -n "$PAPER_OUT/suite.lock" true; then
    echo '本轮已有进程运行，未重复启动。'
    exit 0
fi
nohup bash "$code/research/paper_visuals_v2/launch.sh" \
    >> "$PAPER_OUT/queue.nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PAPER_OUT/suite.pid"
printf '已提交 PID=%s\n输出目录：%s\n总日志：%s/queue.nohup.log\n' "$pid" "$PAPER_OUT" "$PAPER_OUT"
printf 'GPU空闲与正式审计通过后串行执行；完成后保留服务器，等待下载和本地校验。\n'
