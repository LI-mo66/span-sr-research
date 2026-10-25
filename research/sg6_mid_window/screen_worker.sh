#!/usr/bin/env bash
# One nohup worker, two sequential candidates, one guarded final release.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
group=${1:?Experiment group required}
reference=${2:?Fixed SG6 persistent archive required}
[[ "$group" =~ ^[A-Za-z0-9_-]+$ ]]
[[ $(uname -s) == Linux ]]
[[ -z $(git status --porcelain) ]]
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
[[ -x "$python_bin" ]]
backup_root=${BACKUP_ROOT:?Persistent backup root required}
[[ ${AUTO_RELEASE:-1} == 0 || ${AUTO_RELEASE:-1} == 1 ]]
"$python_bin" -c 'from pathlib import Path; import sys; assert Path(sys.argv[1]).resolve().is_relative_to(Path("/home/featurize/work"))' "$backup_root"
mkdir -p "$backup_root"
[[ -w "$backup_root" ]]
if [[ ${AUTO_RELEASE:-1} == 1 ]]; then command -v featurize >/dev/null; fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
group_dir="$PWD/logs/sg6_mid_window/$group"
[[ ! -e "$group_dir" && ! -e "runs/$group-A-s10" && ! -e "runs/$group-C-s10" ]]
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
stage AUDITING_SG6_AND_ORDERED_DATA
"$python_bin" -u research/sg6_mid_window/prepare.py --reference-backup "$reference" --output "$group_dir/audit"
"$python_bin" -m pip freeze > "$group_dir/dependencies.txt"
git archive --format=tar --output="$group_dir/source.tar" HEAD
stage CHECKING_INTERFACES_DISTRIBUTION_OPTIMIZATION_NORMALIZATION
"$python_bin" -u research/sg6_mid_window/verify.py --device cuda --output "$group_dir/engineering-cuda.json"
for kind in A C; do
    stage "CHECKING_${kind}_TRAINER_INTEGRATION"
    "$python_bin" -u research/sg6_mid_window/integration.py --variant "$kind" --output "$group_dir/integration-$kind.json"
    stage "CHECKING_${kind}_REAL_CROPS_AND_LARGEST_IMAGE"
    "$python_bin" -u research/sg6_mid_window/server_smoke.py --config "$group_dir/audit/$kind-config.json" --output "$group_dir/real-data-smoke-$kind.json"
done
stage BENCHMARKING_SG6_A_C
"$python_bin" -u research/sg6_mid_window/benchmark.py --output "$group_dir/resources.json"
stage CHECKING_SPACE_FOR_TWO_RUNS_AND_BACKUP
"$python_bin" -u research/sg6_mid_window/check_space.py --project "$PWD" --group-dir "$group_dir" --destination "$backup_root"
[[ -z $(git status --porcelain) ]]
for kind in A C; do
    stage "TRAINING_${kind}_50000_FROM_SCRATCH"
    "$python_bin" -u research/sg6_mid_window/run.py --config "$group_dir/audit/$kind-config.json" --id "$group-$kind-s10" --screen --screen-updates 50000
    stage "COMPARING_${kind}_WITH_SG6"
    "$python_bin" -u research/sg6_mid_window/compare.py --audit "$group_dir/audit/reuse-audit.json" --run "$PWD/runs/$group-$kind-s10" --variant "$kind" --output "$group_dir/comparison-$kind-50000.json"
done
stage SUMMARIZING_A_C_AND_TARGET_DATASETS
"$python_bin" -u research/sg6_mid_window/summary.py --group-dir "$group_dir"
stage BACKING_UP_BOTH_RUNS
"$python_bin" -u research/sg6_mid_window/finalize.py --project "$PWD" --group-dir "$group_dir" --destination "$backup_root" --receipt "$group_dir/backup-path.txt"
backup=$(cat "$group_dir/backup-path.txt")
[[ -f "$backup/BACKUP_COMPLETE" ]]
printf 'COMPLETE; BOTH_50K_BACKUP_VERIFIED\n' | tee "$group_dir/status.txt" "$backup/release-status.txt"
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
