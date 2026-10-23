#!/usr/bin/env bash
# User-authorized release only after controller success and verified backup.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
python_bin=${1:?Python required}
run_id=${2:?ID required}
reference=${3:?Reference required}
data=${4:?Data root required}
records=${5:?Backup root required}
"$python_bin" -u research/ucan_qk_pair/server.py --id "$run_id" \
  --reference "$reference" --data-root "$data" --backup-root "$records"
backup=$("$python_bin" research/ucan_qk_pair/release_guard.py "logs/ucan_qk_pair/$run_id/status.json")
# The controller and all CUDA children have exited before the release gate.
exec 8>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 8 || { echo 'Release skipped: shared GPU lock held'; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$busy" ]]; then
  printf 'RELEASE_SKIPPED_GPU_BUSY: %s\n' "$busy" | tee "$backup/release-status.txt"
  exit 0
fi
printf 'RELEASE_REQUESTED_AFTER_VERIFIED_SUCCESS\n' | tee "$backup/release-status.txt"
sync
if featurize instance release < /dev/null; then
  printf 'release_command_exit_code=0\n' >> "$backup/release-status.txt"
else
  rc=$?
  printf 'RELEASE_FAILED exit_code=%s\n' "$rc" | tee -a "$backup/release-status.txt"
  sync
  exit 73
fi
