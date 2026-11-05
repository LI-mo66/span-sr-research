#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ -z $(git status --porcelain) ]]
[[ ! -e logs/multiscale-manager && ! -e logs/multiscale50k ]]
mkdir -p logs/multiscale-manager
nohup bash research/multiscale_f0/queue.sh > logs/multiscale-manager/nohup.log 2>&1 < /dev/null &
echo "$!" > logs/multiscale-manager/pid.txt
printf 'PID=%s\nLog=%s/logs/multiscale-manager/nohup.log\nNo automatic shutdown.\n' "$!" "$PWD"
