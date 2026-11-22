#!/usr/bin/env bash
set -Eeuo pipefail
cd "$1"
stage=$2
x3stage="$PWD/logs/x3-clip01-175k-210k-12"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE CUDA_VISIBLE_DEVICES

trap 'rc=$?; if ((rc!=0)); then
    printf "FAILED exit=%s; READ_NOHUP_LOG\n" "$rc" > "$stage/status.txt"
fi' EXIT

mkdir -p /home/featurize/span-x4-runtime/logs
exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock

echo WAITING_FOR_X3_210K > "$stage/status.txt"
echo "等待当前任务释放GPU锁……"
while ! flock -n 9; do
    sleep 30
done

test "$(cat "$x3stage/status.txt")" = COMPLETE_X3_CLIP01_210K_NO_SHUTDOWN
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "GPU_BUSY: $busy"; exit 75; }

[[ $(git rev-parse HEAD) == 0748bd60c1eaddda4e0db7f01ba768a0f2a22be3 ]]
[[ -z $(git status --porcelain) ]]

/environment/miniconda3/bin/python3 - "$stage" <<'PY'
import hashlib
import sys
from pathlib import Path

stage = Path(sys.argv[1])
path = Path("runs/F0-X2-CLIP01-U300K-09/latest.pt")
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == (stage / "source-300k-sha256.txt").read_text().strip()
print("PASS：等待期间×2起始断点保持一致", flush=True)
PY

echo TRAINING_X2_CLIP01_300K_TO_400K > "$stage/status.txt"

/environment/miniconda3/bin/python3 -u - "$stage/continue.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
code = compile(path.read_bytes(), str(path), "exec")
sys.argv = [str(path)]
exec(code, {"__name__": "__main__", "__file__": str(path)})
PY

/environment/miniconda3/bin/python3 - "$stage" <<'PY'
import json
import math
import sys
from pathlib import Path
import torch

stage = Path(sys.argv[1])
run = Path.cwd() / "runs/F0-X2-CLIP01-U400K-13"
state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
assert state["update"] == 400000 and state["epoch"] == 400
assert state["scheduler"]["last_epoch"] == 400000
assert state["scheduler"]["_last_lr"] == [5e-5]
assert all(g["lr"] == 5e-5 for g in state["optimizer"]["param_groups"])
assert not (run / "ANOMALY.json").exists()

rows = [
    json.loads(line) for line in (run / "epochs.jsonl").open()
    if line.strip()
]
assert [r["update"] for r in rows] == list(range(1000, 400001, 1000))
clips = [
    json.loads(line) for line in (stage / "clip-updates.jsonl").open()
    if line.strip()
]
assert [r["update"] for r in clips] == list(range(300001, 400001))
assert all(r["lr_used"] == [1e-4] for r in clips)
assert all(math.isfinite(r["mean_l1"]) for r in clips)
selected = min(
    rows,
    key=lambda row: (-row["metrics"]["DIV2K_valid"]["psnr"], row["update"])
)
summary = {
    "run": str(run),
    "update": state["update"],
    "optimizer_lr": [g["lr"] for g in state["optimizer"]["param_groups"]],
    "clip_limit": 0.1,
    "clip_trigger_count": sum(r["clip_applied"] for r in clips),
    "updates": len(clips),
    "largest_training_l1": max(clips, key=lambda r: r["mean_l1"]),
    "selected_update": selected["update"],
    "selected_metrics": selected["metrics"],
    "fixed_400k_metrics": rows[-1]["metrics"],
    "recent_evaluations": [
        {"update": r["update"], "metrics": r["metrics"]}
        for r in rows if r["update"] >= 391000
    ],
}
(stage / "stage-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PASS：400K完整断点、调度及评测记录检查", flush=True)
PY

echo COMPLETE_X2_CLIP01_400K_NO_SHUTDOWN > "$stage/status.txt"
