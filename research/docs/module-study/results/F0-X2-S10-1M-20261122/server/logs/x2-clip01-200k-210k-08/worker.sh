#!/usr/bin/env bash
set -Eeuo pipefail
cd "$1"
stage=$2
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE

trap 'rc=$?; if ((rc != 0)); then
    printf "FAILED exit=%s; READ_NOHUP_LOG\n" "$rc" > "$stage/status.txt"
fi' EXIT

echo CHECKING_GPU_AND_200K_STATE > "$stage/status.txt"
mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -n 9 || { echo GPU_LOCK_BUSY; exit 75; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU_BUSY: $busy"; exit 75; }

[[ $(git rev-parse HEAD) == 0748bd60c1eaddda4e0db7f01ba768a0f2a22be3 ]]
[[ -z $(git status --porcelain) ]]

echo TRAINING_X2_CLIP01_200K_TO_210K > "$stage/status.txt"
/environment/miniconda3/bin/python3 -u "$stage/continue.py"

/environment/miniconda3/bin/python3 - "$stage" <<'PY'
import json
import math
import sys
from pathlib import Path
import torch

stage = Path(sys.argv[1])
root = Path.cwd()
run = root / "runs/F0-X2-CLIP01-U210K-08"
state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
assert state["update"] == 210000 and state["epoch"] == 210
assert state["scheduler"]["last_epoch"] == 210000
assert state["scheduler"]["_last_lr"] == [1e-4]
assert all(g["lr"] == 1e-4 for g in state["optimizer"]["param_groups"])
assert not (run / "ANOMALY.json").exists()

rows = [
    json.loads(line)
    for line in (run / "epochs.jsonl").open()
    if line.strip()
]
assert [r["update"] for r in rows] == list(range(1000, 210001, 1000))
selected = [r for r in rows if 200000 <= r["update"] <= 210000]
clips = [
    json.loads(line)
    for line in (stage / "clip-updates.jsonl").open()
    if line.strip()
]
assert [r["update"] for r in clips] == list(range(200001, 210001))
assert all(r["lr_used"] == [1e-4] for r in clips)
assert all(math.isfinite(r["mean_l1"]) for r in clips)

report = {
    "run": str(run),
    "update": state["update"],
    "lr": [g["lr"] for g in state["optimizer"]["param_groups"]],
    "clip_limit": 0.1,
    "clip_trigger_count": sum(r["clip_applied"] for r in clips),
    "updates": len(clips),
    "largest_training_l1": max(clips, key=lambda r: r["mean_l1"]),
    "evaluations": [
        {"update": r["update"], "metrics": r["metrics"]} for r in selected
    ],
}
(stage / "stage-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS：210K断点、学习率、训练和评测记录检查通过", flush=True)
PY

echo COMPLETE_X2_CLIP01_210K_NO_SHUTDOWN > "$stage/status.txt"
