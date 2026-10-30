#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
cap=${1:?Stage cap required}
requested=$cap
if [[ "$requested" == check ]]; then cap=100000;export CHECK_ONLY=1;fi
case "$cap" in 100000|250000|500000|750000|1000000) ;; *) echo 'Unregistered stage';exit 1;; esac
[[ -z $(git status --porcelain) ]]
manager="$PWD/logs/f0-long-stage-$requested"
[[ ! -e "$manager" ]] || { echo 'Stage directory exists; refusing duplicate launch';exit 1; }
[[ ${AUTO_RELEASE:-1} == 0 || ${AUTO_RELEASE:-1} == 1 ]]
mkdir -p "$manager"
nohup bash research/f0_long/stage_worker.sh "$cap" "$manager" > "$manager/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$manager/worker.pid"
printf '后台PID：%s\n日志：%s/nohup.log\n' "$!" "$manager"
