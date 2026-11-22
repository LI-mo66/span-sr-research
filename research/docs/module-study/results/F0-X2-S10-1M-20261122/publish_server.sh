#!/usr/bin/env bash
set -euo pipefail
project=${1:-/home/featurize/work/span-x4-research/code/span-f0-x2x3-screen-01}
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
branch=records/f0-x2-1m-20261122
archive=research/docs/module-study/results/F0-X2-S10-1M-20261122
root=/home/featurize/work/span-x4-research/records
mkdir -p "$root"
delivery=$(mktemp -d "$root/github-x2-1m-XXXXXXXX")
printf '独立Git上传目录 %s\n' "$delivery"
git clone --single-branch --branch "$branch" \
    https://github.com/LI-mo66/span-sr-research.git "$delivery/repo"
cd "$delivery/repo"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python_bin" "$archive/collect_server.py" --project "$project"
git add -- "$archive/server"
git -c user.name='LI-mo66' -c user.email='LI-mo66@users.noreply.github.com' \
    commit -m 'Archive verified x2 1M server metadata histories and per-image metrics'
printf '已创建本地提交，开始上传分支 %s\n' "$branch"
git push origin "HEAD:refs/heads/$branch"
printf '上传完成 https://github.com/LI-mo66/span-sr-research/tree/%s/%s/server\n' "$branch" "$archive"
