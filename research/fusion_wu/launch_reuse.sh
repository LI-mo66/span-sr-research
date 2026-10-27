#!/usr/bin/env bash
# Launch exactly one isolated fusion screen; automatic shutdown is disabled.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]]
[[ -z $(git status --porcelain) ]]
manager="$PWD/logs/fusion-WU-launch"
if [[ -e "$manager" || -e logs/fusion-WU || -e runs/FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01 ]]; then
    echo 'Existing fusion logs or run found; refusing duplicate launch.'
    exit 1
fi
mkdir -p "$manager"
nohup bash research/fusion_wu/worker_reuse.sh "$manager" > "$manager/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$manager/worker.pid"
printf '后台PID：%s\n日志：%s/nohup.log\n' "$!" "$manager"
printf 'Automatic shutdown is disabled.\n'
