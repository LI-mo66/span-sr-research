import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch

root = Path.cwd()
stage = Path(sys.argv[1])
original = root / "runs/F0-X3-CLIP01-U300K-14"
checkpoint = original / "epoch00295_u295000.pt"
seed_run = root / "runs/F0-X3-CLIP05-TEST-SEED295K-18"
target = root / "runs/F0-X3-CLIP05-TEST-U300K-18"
for path in (seed_run, target):
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)

manifest = json.loads((original / "manifest.json").read_text())
cfg = json.loads((root / "logs/multiscale50k/x3-F0.json").read_text())
assert manifest["config"] == cfg
assert cfg["arm"] == "F0" and cfg["model"]["upscale"] == 3
assert cfg["seed"] == 10 and cfg["self_ensemble"] is True
assert cfg["hr_patch"] == 192 and cfg["crop_border"] == 3

state = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert state["update"] == 295000 and state["epoch"] == 295
assert state["fingerprint"] == manifest["fingerprint"]
scheduler = state["scheduler"]
optimizer = state["optimizer"]
assert scheduler["last_epoch"] == 295000
assert scheduler["_last_lr"] == [1e-4]
assert scheduler["base_lrs"] == [2e-4]
assert scheduler["gamma"] == 0.5
assert dict(scheduler["milestones"]) == {
    200000: 1, 400000: 1, 600000: 1, 800000: 1,
}
assert len(optimizer["param_groups"]) == 1 and optimizer["state"]
assert optimizer["param_groups"][0]["lr"] == 1e-4
assert all(state[key] is not None for key in (
    "python_rng", "numpy_rng", "torch_rng", "cuda_rng",
))
for tensors in (state["model"], *optimizer["state"].values()):
    assert all(
        bool(torch.isfinite(value).all())
        for value in tensors.values()
        if torch.is_tensor(value) and value.is_floating_point()
    ), "295K模型或Adam状态含非有限值"

assert shutil.disk_usage(root).free >= 3 * 1024**3, "可用空间不足3 GiB"

previous = root / "logs/x3-clip01-210k-300k-14/continue.py"
source = previous.read_text()
changes = (
    (
        'SOURCE = ROOT / "runs/F0-X3-CLIP01-U210K-12"',
        'SOURCE = ROOT / "runs/F0-X3-CLIP05-TEST-SEED295K-18"',
    ),
    (
        'RUNID = "F0-X3-CLIP01-U300K-14"',
        'RUNID = "F0-X3-CLIP05-TEST-U300K-18"',
    ),
    ('assert state["update"] == 210000',
     'assert state["update"] == 295000'),
    ('assert state["scheduler"]["last_epoch"] == 210000',
     'assert state["scheduler"]["last_epoch"] == 295000'),
    (
        'assert all(g["lr"] == 1e-4 for g in state["optimizer"]["param_groups"])',
        'assert all(g["lr"] == 5e-5 for g in state["optimizer"]["param_groups"])',
    ),
    ('assert state["scheduler"]["_last_lr"] == [1e-4]',
     'assert state["scheduler"]["_last_lr"] == [5e-5]'),
)
for old, new in changes:
    if source.count(old) != 1:
        raise RuntimeError(f"续训入口与预期不一致：{old}")
    source = source.replace(old, new)

source = source.replace(
    "X3 continuation210K to300K; global clip0.1; original milestones",
    "X3 recovery295K to300K; global clip0.1; early LR reduction to5e-5",
)
assert "previous.STOP = 300000" in source
assert "previous.train(3)" in source
assert "CLIP_LIMIT = 0.1" in source
assert "F.l1_loss = original_l1" in source
entry_changes = (
    (
        'assert all(g["lr"] == 5e-5 for g in state["optimizer"]["param_groups"])',
        'assert all(g["lr"] == 1e-4 for g in state["optimizer"]["param_groups"])',
    ),
    ('assert state["scheduler"]["_last_lr"] == [5e-5]',
     'assert state["scheduler"]["_last_lr"] == [1e-4]'),
    ("CLIP_LIMIT = 0.1", "CLIP_LIMIT = 0.5"),
    (
        "X3 recovery295K to300K; global clip0.1; early LR reduction to5e-5",
        "X3 test295K to300K; global clip0.5; original LR1e-4 and milestones",
    ),
)
for old, new in entry_changes:
    if source.count(old) != 1:
        raise RuntimeError(f"生成的训练入口与预期不一致：{old}")
    source = source.replace(old, new)
entry = stage / "continue.py"
compile(source, str(entry), "exec")
entry.write_text(source)

# Create a separate restart checkpoint from the original 295K state.
with checkpoint.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
# Keep the original optimizer and scheduler state.

seed_run.mkdir()
shutil.copy2(original / "manifest.json", seed_run / "manifest.json")
historical_loss_gaps = []
for name, stride in (("epochs.jsonl", 1000), ("loss_steps.jsonl", 1)):
    expected = stride
    with (original / name).open() as src, (seed_run / name).open("x") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            update = row["update"]
            if update > 295000:
                break
            if update != expected:
                # Permit forward gaps entirely before the 1000-step window.
                if name != "loss_steps.jsonl" or not expected < update <= 294001:
                    raise RuntimeError(
                        f"{name}: expected update {expected}, found {update}"
                    )
                historical_loss_gaps.append({
                    "first_missing": expected,
                    "last_missing": update - 1,
                    "count": update - expected,
                })
            if name == "loss_steps.jsonl":
                assert math.isfinite(row["train_l1"])
            dst.write(json.dumps(row) + "\n")
            expected = update + stride
    assert expected == 295000 + stride, (name, expected)

shutil.copy2(checkpoint, seed_run / "latest.tmp")
with (seed_run / "latest.tmp").open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
(seed_run / "latest.tmp").replace(seed_run / "latest.pt")
(stage / "provenance.json").write_text(json.dumps({
    "original_checkpoint": str(checkpoint),
    "original_sha256": digest,
    "historical_loss_gaps": historical_loss_gaps,
    "start_update": 295000,
    "stop_update": 300000,
    "original_lr": 1e-4,
    "recovery_lr": 1e-4,
    "clip_limit": 0.5,
}, indent=2) + "\n")
print("PASS：295K完整断点通过验收，独立恢复起点已生成")

print("历史损失记录缺口", json.dumps(historical_loss_gaps), flush=True)
