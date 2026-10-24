#!/usr/bin/env bash
# Read-only trained-weight diagnostic; outputs only to a new dedicated job.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
candidate=${1:?Candidate backup required}
reference=${2:?SG6 backup required}
job=${3:?New persistent diagnostic output required}
work=${4:?New fast scratch directory required}
for program in nvidia-smi flock featurize; do command -v "$program" >/dev/null; done
mkdir -p /home/featurize/span-x4-runtime/logs
exec 8>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 8 || { echo 'Other GPU queue active'; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "Other GPU processes present: $busy"; exit 75; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" -u research/ucan_qk_pair/mechanism_audit.py \
 --candidate "$candidate" --reference "$reference" --output "$job" --work "$work"
"$python_bin" - "$job" <<'PY'
from pathlib import Path
import hashlib,json,sys
job=Path(sys.argv[1]).resolve()
assert job.is_relative_to(Path('/home/featurize/work'))
assert json.loads((job/'STATUS.json').read_text())['success'] is True
expected=(job/'diagnostic-results.sha256').read_text().strip()
assert hashlib.sha256((job/'diagnostic-results.zip').read_bytes()).hexdigest()==expected
print('Verified diagnostic archive:',job/'diagnostic-results.zip',flush=True)
PY
if [[ ${AUTO_RELEASE:-1} == 1 ]]; then
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [[ -z "$busy" ]] || { echo "Instance retained: other GPU processes: $busy"; exit 0; }
  printf 'RELEASE_REQUESTED_AFTER_DIAGNOSTIC_ARCHIVE_VERIFIED\n' > "$job/release-status.txt"
  sync
  if featurize instance release < /dev/null; then
    printf 'release_command_exit_code=0\n' >> "$job/release-status.txt"
  else
    rc=$?;printf 'release_command_exit_code=%s\n' "$rc" >> "$job/release-status.txt";exit 73
  fi
fi
