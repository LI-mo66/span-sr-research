import json
import math
import sys
from pathlib import Path

import torch

stage = Path(sys.argv[1])
original = Path("runs/F0-X3-CLIP01-U300K-14")
run = Path("runs/F0-X3-CLIP01-RECOVERY-U300K-17")

state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
assert state["update"] == 300000 and state["epoch"] == 300
assert state["scheduler"]["last_epoch"] == 300000
assert state["scheduler"]["_last_lr"] == [5e-5]
assert [g["lr"] for g in state["optimizer"]["param_groups"]] == [5e-5]
assert not (run / "ANOMALY.json").exists()

def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

clips = list(records(stage / "clip-updates.jsonl"))
assert [row["update"] for row in clips] == list(range(295001, 300001))
assert all(
    row["lr_used"] == [5e-5]
    and row["clip_limit"] == 0.1
    and math.isfinite(row["mean_l1"])
    and math.isfinite(row["gradient_norm_before_clip"])
    for row in clips
)

def evaluations(folder):
    rows = [
        row for row in records(folder / "epochs.jsonl")
        if 295000 <= row["update"] <= 300000
    ]
    assert [row["update"] for row in rows] == list(range(295000, 300001, 1000))
    return rows

old_rows = evaluations(original)
new_rows = evaluations(run)
assert old_rows[0]["metrics"] == new_rows[0]["metrics"]
comparisons = []
for old, new in zip(old_rows, new_rows):
    delta = {}
    for name, score in new["metrics"].items():
        assert all(math.isfinite(score[key]) for key in ("psnr", "ssim"))
        delta[name] = score["psnr"] - old["metrics"][name]["psnr"]
    comparisons.append({
        "update": new["update"],
        "original_psnr": {
            name: score["psnr"] for name, score in old["metrics"].items()
        },
        "recovery_psnr": {
            name: score["psnr"] for name, score in new["metrics"].items()
        },
        "delta_recovery_vs_original": delta,
    })

summary = {
    "update": state["update"],
    "lr": state["scheduler"]["_last_lr"],
    "updates": len(clips),
    "clip_trigger_count": sum(row["clip_applied"] for row in clips),
    "max_train_l1": max(
        (row["mean_l1"], row["update"]) for row in clips
    ),
    "max_preclip_gradient_norm": max(
        (row["gradient_norm_before_clip"], row["update"]) for row in clips
    ),
    "evaluations": comparisons,
}
(stage / "stage-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PASS：300K恢复试验完成，已保存同起点PSNR对照")
