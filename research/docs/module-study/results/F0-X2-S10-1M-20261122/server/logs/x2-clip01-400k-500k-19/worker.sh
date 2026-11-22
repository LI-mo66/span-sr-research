#!/usr/bin/env bash
set -Eeuo pipefail

cd "$1"
stage=$2
python_bin=$3
x3_stage=$4
previous="$PWD/logs/x2-clip01-300k-400k-13"

export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONOPTIMIZE MS_VERIFY_CASE CUDA_VISIBLE_DEVICES

phase=WAITING_FOR_X3_CLIP05
trap 'rc=$?; if ((rc != 0)); then
    printf "FAILED phase=%s exit=%s\n" "$phase" "$rc" | tee "$stage/status.txt"
fi' EXIT

printf '%s\n' "$phase" > "$stage/status.txt"
echo "等待×3的clip=0.5试验退出"
while true; do
    x3_status=$(cat "$x3_stage/status.txt" 2>/dev/null || true)
    case "$x3_status" in
        COMPLETE_X3_CLIP05_TEST_300K_NO_SHUTDOWN|FAILED*)
            printf '×3已进入终止状态：%s\n' "$x3_status"
            break
            ;;
    esac
    sleep 30
done

phase=CHECKING_X2_400K
printf '%s\n' "$phase" > "$stage/status.txt"

exec 9>>/home/featurize/span-x4-runtime/logs/server-gpu.lock
flock -x 9
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$busy" ]]; then
    printf 'GPU_BUSY: %s\n' "$busy"
    exit 75
fi

test "$(cat "$previous/status.txt")" = COMPLETE_X2_CLIP01_400K_NO_SHUTDOWN
test "$(git rev-parse HEAD)" = 0748bd60c1eaddda4e0db7f01ba768a0f2a22be3
test -z "$(git status --porcelain)"
test -d datasets/image_cache

CUDA_VISIBLE_DEVICES="" "$python_bin" - "$stage" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

stage = Path(sys.argv[1])
run = Path("runs/F0-X2-CLIP01-U400K-13")
target = Path("runs/F0-X2-CLIP01-U500K-19")
assert not target.exists() and not target.is_symlink(), target
assert not (run / "ANOMALY.json").exists()

checkpoint = run / "latest.pt"
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
scheduler = state["scheduler"]
assert state["update"] == 400000 and state["epoch"] == 400
assert scheduler["last_epoch"] == 400000
assert scheduler["_last_lr"] == [5e-5]
assert scheduler["base_lrs"] == [2e-4]
assert scheduler["gamma"] == 0.5
assert dict(scheduler["milestones"]) == {
    200000: 1, 400000: 1, 600000: 1, 800000: 1
}
assert state["optimizer"]["state"]
assert [g["lr"] for g in state["optimizer"]["param_groups"]] == [5e-5]
assert all(state[key] is not None for key in (
    "python_rng", "numpy_rng", "torch_rng", "cuda_rng"
))

manifest = json.loads((run / "manifest.json").read_text())
cfg = json.loads(Path("logs/multiscale50k/x2-F0.json").read_text())
assert state["fingerprint"] == manifest["fingerprint"]
assert manifest["config"] == cfg
assert cfg["arm"] == "F0" and cfg["model"]["upscale"] == 2
assert cfg["seed"] == 10 and cfg["self_ensemble"] is True
assert cfg["hr_patch"] == 128 and cfg["crop_border"] == 2

required = (
    sum((run / name).stat().st_size for name in (
        "latest.pt", "manifest.json", "epochs.jsonl", "loss_steps.jsonl"
    ))
    + 105 * checkpoint.stat().st_size
    + 5 * 1024**3
)
free = shutil.disk_usage(stage).free
print(f"可用空间 {free / 1024**3:.2f} GiB，预计需要 {required / 1024**3:.2f} GiB")
assert free >= required, "磁盘空间不足"

with checkpoint.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
(stage / "provenance.json").write_text(json.dumps({
    "source_checkpoint": str(checkpoint.resolve()),
    "source_sha256": digest,
    "start_update": 400000,
    "stop_update": 500000,
    "lr": 5e-5,
    "clip_limit": 0.1,
}, indent=2) + "\n")
print("PASS：×2的400K完整断点和原调度通过验收")
PY

phase=TRAINING_X2_CLIP01_400K_TO_500K
printf '%s\n' "$phase" > "$stage/status.txt"
"$python_bin" -u - "$stage/continue.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
code = compile(path.read_bytes(), str(path), "exec")
sys.argv = [str(path)]
exec(code, {"__name__": "__main__", "__file__": str(path)})
PY

phase=VERIFYING_X2_500K
printf '%s\n' "$phase" > "$stage/status.txt"

CUDA_VISIBLE_DEVICES="" "$python_bin" - "$stage" <<'PY'
import json
import math
import sys
from pathlib import Path
from statistics import mean

import torch

stage = Path(sys.argv[1])
run = Path("runs/F0-X2-CLIP01-U500K-19")
state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
assert state["update"] == 500000 and state["epoch"] == 500
assert state["scheduler"]["last_epoch"] == 500000
assert state["scheduler"]["_last_lr"] == [5e-5]
assert [g["lr"] for g in state["optimizer"]["param_groups"]] == [5e-5]
assert not (run / "ANOMALY.json").exists()

with (run / "epochs.jsonl").open() as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
assert [row["update"] for row in rows] == list(range(1000, 500001, 1000))
new_rows = [row for row in rows if row["update"] > 400000]
for row in new_rows:
    assert all(
        math.isfinite(score[key])
        for score in row["metrics"].values()
        for key in ("psnr", "ssim")
    )

count = clipped = 0
max_loss = max_grad = (0.0, None)
with (stage / "clip-updates.jsonl").open() as stream:
    for line in stream:
        if not line.strip():
            continue
        row = json.loads(line)
        count += 1
        assert row["update"] == 400000 + count
        assert row["lr_used"] == [5e-5] and row["clip_limit"] == 0.1
        loss = row["mean_l1"]
        grad = row["gradient_norm_before_clip"]
        assert math.isfinite(loss) and math.isfinite(grad)
        clipped += int(row["clip_applied"])
        if loss > max_loss[0]:
            max_loss = (loss, row["update"])
        if grad > max_grad[0]:
            max_grad = (grad, row["update"])
assert count == 100000

selected = min(
    new_rows,
    key=lambda row: (-row["metrics"]["DIV2K_valid"]["psnr"], row["update"]),
)
selected_path = run / f"epoch{selected['epoch']:05d}_u{selected['update']}.pt"
assert selected_path.is_file()
baseline = next(row for row in rows if row["update"] == 400000)
final = rows[-1]
summary = {
    "run": str(run.resolve()),
    "update": 500000,
    "lr": [5e-5],
    "clip_limit": 0.1,
    "updates": count,
    "clip_trigger_count": clipped,
    "clip_fraction": clipped / count,
    "max_train_l1_value_and_update": max_loss,
    "max_preclip_gradient_norm_value_and_update": max_grad,
    "selected_update_after_400k": selected["update"],
    "selected_checkpoint": str(selected_path.resolve()),
    "selected_metrics": selected["metrics"],
    "fixed_500k_metrics": final["metrics"],
    "fixed_500k_delta_psnr_vs_400k": {
        name: score["psnr"] - baseline["metrics"][name]["psnr"]
        for name, score in final["metrics"].items()
    },
    "mean_psnr_491k_to_500k": {
        name: mean(row["metrics"][name]["psnr"] for row in new_rows[-10:])
        for name in final["metrics"]
    },
}
(stage / "stage-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PASS：×2的500K断点、裁剪记录和评测通过验收")
PY

echo COMPLETE_X2_CLIP01_500K_NO_SHUTDOWN > "$stage/status.txt"
