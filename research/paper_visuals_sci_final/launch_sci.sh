#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
: "${PYTHON_BIN:=/environment/miniconda3/bin/python3}"
: "${PAPER_BASE:=/home/featurize/work/span-x4-research}"
: "${PAPER_PROJECT:=$PAPER_BASE/code/span-sg6-f0-long-x8-1m}"
: "${PAPER_ROOT:=$PAPER_BASE/records/paper_visuals_20261126}"
: "${PAPER_OLD_OUT:=$PAPER_ROOT/step10-paper-suite-v2-01}"
: "${PAPER_OUT:=$PAPER_ROOT/step11-paper-sci-final-01}"
: "${PAPER_DATA:=/home/featurize/datasets/benchmark}"
export PAPER_BASE PAPER_PROJECT PAPER_ROOT PAPER_OLD_OUT PAPER_OUT PAPER_DATA
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
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader || true)
[[ -z "$busy" ]] || { echo "GPU_BUSY: $busy"; exit 75; }

stage=STARTING
trap 'code=$?; printf "STOPPED stage=%s exit=%s; no shutdown\n" "$stage" "$code" > "$PAPER_OUT/status.txt"' ERR

TOTAL_STEPS=28
CUR_STEP=0

step() {
    CUR_STEP=$((CUR_STEP + 1))
    stage=$1
    local marker=$2
    shift 2
    printf '%s\n' "$stage" > "$PAPER_OUT/status.txt"
    if [[ -s "$PAPER_OUT/$marker" ]]; then
        echo "[$CUR_STEP/$TOTAL_STEPS] SKIP_COMPLETED $stage"
    else
        echo "[$CUR_STEP/$TOTAL_STEPS] START $stage ($(date '+%H:%M:%S'))"
        if [[ -e "$PAPER_OUT/$stage.log" ]]; then
            mv -n "$PAPER_OUT/$stage.log" "$PAPER_OUT/$stage.failed-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
        fi
        "$PYTHON_BIN" -u "$@" 2>&1 | tee "$PAPER_OUT/$stage.log"
        test -s "$PAPER_OUT/$marker"
        echo "[$CUR_STEP/$TOTAL_STEPS] PASS $stage ($(date '+%H:%M:%S'))"
    fi
}

# Task 3: 50K Ablation Convergence Curve (fast CPU, copies 5 epochs.jsonl into portable assets)
step convergence fig_ablation_convergence_50k.json plot_convergence.py

# Task 1 & Task 7: Unified 5-benchmark single-pass inference (reuses existing PNGs; computes B100/Set5 + new LBNet/SMFANet/SMFANet+)
step ext-Bicubic      external-bicubic.json                 compute_sci.py external --name Bicubic
step ext-SPAN52       external-span52_official_single.json  compute_sci.py external --name SPAN52
step ext-LBNet        external-lbnet_single.json            compute_sci.py external --name LBNet
step ext-DDistill-SR  external-ddistill_sr_single.json      compute_sci.py external --name DDistill-SR
step ext-SMFANet      external-smfanet_single.json          compute_sci.py external --name SMFANet
step ext-SMFANetPlus  external-smfanet_plus_single.json     compute_sci.py external --name SMFANet+
step ext-IMDN         external-imdn_single.json             compute_sci.py external --name IMDN
step ext-SwinIR-light external-swinir_light_single.json     compute_sci.py external --name SwinIR-light
step ext-NGswin       external-ngswin_single.json           compute_sci.py external --name NGswin
step ext-Ours         external-ours_single.json             compute_sci.py external --name Ours

# Task 2: 50K Historical 48K Ablation Models on Urban100 img004 & img011
for model in B0 SG6 W U F0; do
    step "ablation-$model" "ablation-$model.json" compute_sci.py ablation --name "$model"
done

# Task 4: Official LAM + DI (NGswin auto-skipped cleanly if gradient/memory incompatible)
step lam-SPAN52       lam-SPAN52.json       compute_sci.py lam --name SPAN52
step lam-SwinIR-light lam-SwinIR-light.json compute_sci.py lam --name SwinIR-light
step lam-NGswin       lam-NGswin.json       compute_sci.py lam --name NGswin --allow-skip
step lam-Ours         lam-Ours.json         compute_sci.py lam --name Ours

# Task 5: Ours Intermediate Feature Evolution + Gate G Hooks
step features fig6_stats.json compute_sci.py features

# Task 6: Unified 1280x720 RTX 4090 Complexity, Latency & Peak Memory Benchmark
step efficiency efficiency_720p_rtx4090.json compute_sci.py efficiency --warmup 20 --repeats 100

# Render all upgraded SCI figures (dual 8-scene Fig3a/3b, 6-col Fig4, Fig5 LAM, 6-col Fig6)
step render-fig3 fig3_v2_manifest.json render_sci.py fig3
step render-fig4 fig4_manifest.json    render_sci.py fig4
step render-fig5 fig5_manifest.json    render_sci.py fig5
step render-fig6 fig6_manifest.json    render_sci.py fig6

# Task 7: Merge 5-benchmark metrics, update efficiency Avg5, build portable review ZIP + SHA256SUMS
step merge MERGE_COMPLETE.json merge_and_pack.py merge

# Full server backup archive (skip if SKIP_FULL_ARCHIVE=1)
if [[ "${SKIP_FULL_ARCHIVE:-0}" != "1" ]]; then
    step archive backup-path.txt merge_and_pack.py archive
fi

printf 'Completed SCI final suite (Tasks 1-7) and verified server backup. Download review ZIP and verify locally before releasing RTX 4090.\n' > "$PAPER_OUT/COMPUTE_AND_SERVER_BACKUP_COMPLETE"
printf 'COMPLETE_PENDING_LOCAL_DOWNLOAD_AND_TNR_FINAL_RENDER\n' > "$PAPER_OUT/status.txt"
echo COMPLETE_PENDING_LOCAL_DOWNLOAD_AND_TNR_FINAL_RENDER
