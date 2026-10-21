#!/usr/bin/env bash
# One foreground worker under nohup; never launches a second detached child.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
group=${1:?Experiment group required}
reference=${2:?SG6 persistent archive required}
[[ "$group" =~ ^[A-Za-z0-9_-]+$ ]]
[[ $(uname -s) == Linux ]]
[[ -z $(git status --porcelain) ]]
python_bin=${PYTHON_BIN:?Existing isolated Python required}
[[ -x "$python_bin" ]]
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
[[ ${AUTO_RELEASE:-1} == 0 || ${AUTO_RELEASE:-1} == 1 ]]
backup_root=${BACKUP_ROOT:?Persistent backup root required}
"$python_bin" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); assert p.is_relative_to(Path("/home/featurize/work"))' "$backup_root"
mkdir -p "$backup_root"
[[ -w "$backup_root" ]]
if [[ ${AUTO_RELEASE:-1} == 1 ]]; then command -v featurize >/dev/null; fi
run_id="$group-M3-tail-s10"
group_dir="$PWD/logs/sg6_assm_tail/$group"
[[ ! -e "$group_dir" && ! -e "$PWD/runs/$run_id" ]]
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo 'Another research job holds GPU lock'; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU busy: $busy"; exit 75; }
mkdir -p "$group_dir"
exec >> "$group_dir/nohup.log" 2>&1
phase=STARTING
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED phase=%s exit=%s; INSTANCE_RETAINED\n" "$phase" "$rc" > "$group_dir/status.txt"; fi' EXIT
stage() { phase=$1; printf '%s\n' "$phase" | tee "$group_dir/status.txt"; }
stage AUDITING_SG6_AND_DATA
"$python_bin" -u research/sg6_assm_tail/prepare_reuse.py --reference-backup "$reference" --output "$group_dir/audit"
"$python_bin" -m pip freeze > "$group_dir/dependencies.txt"
git archive --format=tar --output="$group_dir/source.tar" HEAD
stage CHECKING_CUDA_KERNEL
"$python_bin" -u research/sg6_assm_tail/probe_cuda.py --backend cuda_isolated --output "$group_dir/kernel-probe.json"
"$python_bin" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="PASS_SMALL_KERNEL"' "$group_dir/kernel-probe.json"
stage CHECKING_MODEL_AND_RESUME
"$python_bin" -u research/sg6_assm_tail/verify.py --device cuda --backend cuda_isolated --output "$group_dir/engineering-cuda.json"
stage CHECKING_THREE_ASPECTS
"$python_bin" -u research/sg6_assm_tail/audit_three_aspects.py --device cuda --backend cuda_isolated --output "$group_dir/three-aspects-cuda.json"
stage CHECKING_TRAINER_INTEGRATION
unset ENGINEERING_BACKEND
"$python_bin" -u research/sg6_assm_tail/integration.py --output "$group_dir/trainer-integration.json"
stage CHECKING_REAL_CROPS_AND_LARGEST_IMAGE
"$python_bin" -u research/sg6_assm_tail/server_smoke.py --config "$group_dir/audit/M2-config.json" --output "$group_dir/real-data-smoke.json"
[[ -z $(git status --porcelain) ]]
stage TRAINING_50000_FROM_SCRATCH
"$python_bin" -u research/sg6_assm_tail/run.py --config "$group_dir/audit/M2-config.json" --id "$run_id" --screen --screen-updates 50000
stage COMPARING_WITH_SG6
"$python_bin" -u research/sg6_assm_tail/compare_reuse.py --audit "$group_dir/audit/reuse-audit.json" --run "$PWD/runs/$run_id" --output "$group_dir/comparison-50000.json"
stage BACKING_UP
"$python_bin" -u research/sg6_assm_tail/finalize_reuse.py --project "$PWD" --group-dir "$group_dir" --run-id "$run_id" --destination "$backup_root" --receipt "$group_dir/backup-path.txt"
backup=$(cat "$group_dir/backup-path.txt")
[[ -f "$backup/BACKUP_COMPLETE" ]]
printf 'COMPLETE; BACKUP_VERIFIED\n' | tee "$group_dir/status.txt" "$backup/release-status.txt"
sync
if [[ ${AUTO_RELEASE:-1} == 1 ]]; then
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$busy" ]]; then
        printf 'COMPLETE; RELEASE_SKIPPED_GPU_BUSY: %s\n' "$busy" | tee "$group_dir/status.txt" "$backup/release-status.txt"
        exit 0
    fi
    printf 'COMPLETE; RELEASE_REQUESTED\n' | tee "$group_dir/status.txt" "$backup/release-status.txt"
    sync
    set +e
    featurize instance release < /dev/null
    rc=$?
    set -e
    printf 'release_command_exit_code=%s\n' "$rc" >> "$backup/release-status.txt"
    if (( rc != 0 )); then
        printf 'RELEASE_FAILED\n' >> "$backup/release-status.txt"
        exit 73
    fi
fi
