#!/usr/bin/env bash
# One independent arm only; never shuts down/releases the instance.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
run_id=${1:-FEAT-SG6-UCAN-IND-V2-50K-01}
reference=${2:-/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269}
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
data=${DATA_ROOT:-/home/featurize/datasets}
records=${BACKUP_ROOT:-/home/featurize/work/span-x4-research/records}
[[ $(uname -s) == Linux ]]
[[ "$run_id" =~ ^FEAT-SG6-UCAN-IND-[A-Za-z0-9_-]+$ ]]
test -x "$python_bin"
test -f "$reference/BACKUP_COMPLETE"
for program in git nohup flock nvidia-smi; do command -v "$program" >/dev/null; done
[[ -z $(git status --porcelain) ]] || { echo 'Source must be committed and clean'; exit 2; }
job="logs/ucan_qk_pair/$run_id"
[[ ! -e "$job" && ! -e "runs/$run_id" ]] || { echo 'Existing outputs preserved; choose a fresh experiment ID'; exit 2; }
mkdir -p "$job"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
nohup "$python_bin" -u research/ucan_qk_pair/server.py --id "$run_id" \
  --reference "$reference" --data-root "$data" --backup-root "$records" \
  > "$job/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$job/pid"
printf 'PID=%s\nLog=%s/%s/nohup.log\nStatus=%s/%s/status.json\n' "$pid" "$PWD" "$job" "$PWD" "$job"
echo 'Background audit/checks submitted. Only independent-v2 50K can train. No automatic shutdown.'
