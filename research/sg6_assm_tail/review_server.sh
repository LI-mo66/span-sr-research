#!/usr/bin/env bash
# Isolated repeatability diagnostic only; never installs, trains 50K or releases.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
id=${1:-REPEAT-REVIEW-03}
[[ "$id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ -z $(git status --porcelain) ]] || { echo 'Source must be clean'; exit 2; }
python_bin=${PYTHON_BIN:-/home/featurize/work/span-x4-research/envs/sg6-assm-torch222-v1/bin/python}
test -x "$python_bin"
group="$PWD/logs/sg6_assm_tail/$id"
[[ ! -e "$group" ]] || { echo 'Prior review preserved; choose a new ID'; exit 2; }
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo 'GPU lock held by another job'; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU busy: $busy"; exit 75; }
mkdir -p "$group"
trap 'rc=$?; if (( rc != 0 )); then printf "REVIEW_PROCESS_FAILED exit=%s\n" "$rc" > "$group/status.txt"; fi' EXIT
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'REVIEW_RUNNING\n' > "$group/status.txt"
"$python_bin" -u research/sg6_assm_tail/review_determinism.py --output "$group/repeat-review.json"
"$python_bin" - "$group" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'repeat-review.json').read_text())
(p/'status.txt').write_text(r['status']+'\n')
print('Diagnostic only:',r['status'],'; no production change, training or release')
PY
