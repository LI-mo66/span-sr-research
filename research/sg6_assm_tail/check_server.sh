#!/usr/bin/env bash
# Existing environment only. No installation, dataset mutation, 50K or release.
set -euo pipefail
cd "$(dirname "$0")/../.."
id=${1:-ISOLATED-CHECK-04}
[[ "$id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ $(uname -s) == Linux ]] || { echo 'Linux CUDA server required'; exit 2; }
[[ -z $(git status --porcelain) ]] || { echo 'Source must be clean'; exit 2; }
python_bin=${PYTHON_BIN:-/environment/miniconda3/bin/python3}
group_dir="$PWD/logs/sg6_assm_tail/$id"
[[ ! -e "$group_dir" ]] || { echo 'Existing check preserved; choose a new check ID'; exit 2; }
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo 'Another research job holds GPU lock'; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU busy: $busy"; exit 75; }
mkdir -p "$group_dir"
trap 'rc=$?; if (( rc != 0 )); then printf "CHECK_FAILED exit=%s\n" "$rc" > "$group_dir/status.txt"; fi' EXIT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
printf 'CHECKING_KERNEL\n' > "$group_dir/status.txt"
"$python_bin" -u research/sg6_assm_tail/probe_cuda.py --output "$group_dir/kernel-probe.json"
status=$("$python_bin" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$group_dir/kernel-probe.json")
if [[ "$status" != PASS_SMALL_KERNEL ]]; then
    printf 'CHECK_STOPPED: %s\n' "$status" | tee "$group_dir/status.txt"
    echo "Send $group_dir/kernel-probe.json; no installation or training attempted."
    exit 0
fi
printf 'CHECKING_MODEL\n' > "$group_dir/status.txt"
"$python_bin" -u research/sg6_assm_tail/verify.py --device cuda --backend cuda_isolated --output "$group_dir/engineering-cuda.json"
printf 'CHECKING_THREE_ASPECTS\n' > "$group_dir/status.txt"
"$python_bin" -u research/sg6_assm_tail/audit_three_aspects.py --device cuda --backend cuda_isolated --output "$group_dir/three-aspects-cuda.json"
"$python_bin" -u research/sg6_assm_tail/resource_smoke.py --output "$group_dir/micro16-cuda.json"
printf 'ENGINEERING_COMPLETE; REAL_DATA_AND_LARGEST_EVAL_PENDING\n' | tee "$group_dir/status.txt"
echo "Check results: $group_dir"
