#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
: "${PYTHON_BIN:=/environment/miniconda3/bin/python3}"
: "${PAPER_BASE:=/home/featurize/work/span-x4-research}"
: "${PAPER_ROOT:=$PAPER_BASE/records/paper_visuals_20261126}"
: "${PAPER_OUT:=$PAPER_ROOT/step10-paper-suite-v2-01}"
export PAPER_BASE PAPER_ROOT PAPER_OUT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$PAPER_OUT"
exec 8>>"$PAPER_OUT/suite.lock"
flock -n 8 || { echo SUITE_ALREADY_RUNNING; exit 75; }
if [[ -e "$PAPER_OUT/COMPUTE_AND_SERVER_BACKUP_COMPLETE" ]]; then
    echo ALREADY_COMPLETED
    exit 0
fi
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU_BUSY: $busy"; exit 75; }
stage=STARTING
trap 'code=$?; printf "STOPPED stage=%s exit=%s; no shutdown\n" "$stage" "$code" > "$PAPER_OUT/status.txt"' ERR
if [[ ${PAPER_FIX_W_ENTRY:-0} == 1 ]]; then
    stage=repair-W-entry-01
    mkdir -p "$PAPER_OUT/$stage"
    printf '%s\n' "$stage" > "$PAPER_OUT/status.txt"
    echo "START $stage"
    "$PYTHON_BIN" -u repair.py audit >> "$PAPER_OUT/$stage.log" 2>&1
    echo "PASS $stage"
fi
step() {
    stage=$1
    local marker=$2
    shift 2
    printf '%s\n' "$stage" > "$PAPER_OUT/status.txt"
    if [[ -s "$PAPER_OUT/$marker" ]]; then
        echo "SKIP_COMPLETED $stage"
    else
        echo "START $stage"
        if [[ -e "$PAPER_OUT/$stage.log" ]]; then
            mv -n "$PAPER_OUT/$stage.log" "$PAPER_OUT/$stage.failed-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
        fi
        "$PYTHON_BIN" -u "$@" > "$PAPER_OUT/$stage.log" 2>&1
        test -s "$PAPER_OUT/$marker"
        echo "PASS $stage"
    fi
}
step preflight preflight.json run.py preflight
step ddistill ddistill-results.json compute.py ddistill
step imdn imdn-results.json compute.py imdn
step fig3 fig3-v2.json render.py fig3
for model in B0 SG6 W U F0; do
    step "ablation-$model" "ablation-$model.json" compute.py ablation --name "$model"
done
step fig4 fig4-v2.json render.py fig4
for model in SPAN52 SwinIR-light Ours; do
    step "lam-$model" "lam-$model.json" compute.py lam --name "$model"
done
step fig5 fig5-v2.json render.py fig5
step features features.json compute.py features
step fig6 fig6-v2.json render.py fig6
step merge MERGE_COMPLETE.json run.py merge
stage=archive
printf '%s\n' "$stage" > "$PAPER_OUT/status.txt"
if [[ ! -s "$PAPER_OUT/backup-path.txt" ]]; then
    "$PYTHON_BIN" -u run.py archive > "$PAPER_OUT/archive.log" 2>&1
fi
printf 'Completed computing figures and verified server backup. Local visual review, download and hash verification pending. No shutdown.\n' > "$PAPER_OUT/COMPUTE_AND_SERVER_BACKUP_COMPLETE"
printf 'COMPLETE_PENDING_LOCAL_REVIEW_AND_DOWNLOAD_NO_SHUTDOWN\n' > "$PAPER_OUT/status.txt"
echo COMPLETE_PENDING_LOCAL_REVIEW_AND_DOWNLOAD_NO_SHUTDOWN
