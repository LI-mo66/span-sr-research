import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean

import torch

ROOT = Path.cwd()
STAGE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "logs/x2-clip01-500k-600k-20"
RUNS = ROOT / "runs"
ID600 = "F0-X2-CLIP01-U600K-20"
ID800 = "F0-X2-CLIP01-U800K-21"
ID1000 = "F0-X2-CLIP01-U1000K-21"
DATASETS = ("DIV2K_valid", "Set5", "Set14", "B100", "Urban100", "Manga109")
LABELS = [label for name in DATASETS for label in (name, name + "_x8")]
SHA600 = "4b25d94f4ad44e11146e0b0cd97044234f1b915ba12fcc8fd93794754a602eca"

def status(message):
    path = STAGE / "status.tmp"
    path.write_text(message + "\n")
    path.replace(STAGE / "status.txt")
    print(message, flush=True)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def checkpoint_check(run, update, lr):
    state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    scheduler = state["scheduler"]
    adam = state["optimizer"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert state["update"] == update and state["epoch"] == update // 1000
    assert state["fingerprint"] == manifest["fingerprint"]
    assert scheduler["last_epoch"] == update
    assert scheduler["_last_lr"] == [lr] and scheduler["base_lrs"] == [2e-4]
    assert scheduler["gamma"] == 0.5
    assert dict(scheduler["milestones"]) == {
        200000: 1, 400000: 1, 600000: 1, 800000: 1
    }
    assert adam["state"] and [g["lr"] for g in adam["param_groups"]] == [lr]
    assert all(state.get(key) is not None for key in (
        "python_rng", "numpy_rng", "torch_rng", "cuda_rng"
    ))
    assert all(
        bool(torch.isfinite(value).all())
        for mapping in (state["model"], *adam["state"].values())
        for value in mapping.values()
        if torch.is_tensor(value) and value.is_floating_point()
    )
    assert not (run / "ANOMALY.json").exists()
    return manifest["config"]

def make_entry(folder, source_id, target_id, start, stop, lr):
    source = (PREVIOUS / "continue.py").read_text(encoding="utf-8")
    changes = (
        ('SOURCE = ROOT / "runs/F0-X2-CLIP01-U500K-19"',
         f'SOURCE = ROOT / "runs/{source_id}"'),
        ('RUNID = "F0-X2-CLIP01-U600K-20"', f'RUNID = "{target_id}"'),
        ('assert state["update"] == 500000', f'assert state["update"] == {start}'),
        ('assert state["scheduler"]["last_epoch"] == 500000',
         f'assert state["scheduler"]["last_epoch"] == {start}'),
        ('assert all(g["lr"] == 5e-5 for g in state["optimizer"]["param_groups"])',
         f'assert all(g["lr"] == {lr!r} for g in state["optimizer"]["param_groups"])'),
        ('assert state["scheduler"]["_last_lr"] == [5e-5]',
         f'assert state["scheduler"]["_last_lr"] == [{lr!r}]'),
        ('previous.NEW_CAP = previous.NEW_CAP.replace("300000", "600000")',
         f'previous.NEW_CAP = previous.NEW_CAP.replace("300000", "{stop}")'),
        ('previous.STOP = 600000', f'previous.STOP = {stop}'),
    )
    for old, new in changes:
        if source.count(old) != 1:
            raise RuntimeError(f"600K入口与预期不一致 {old}")
        source = source.replace(old, new)
    source = source.replace(
        "X2 continuation500K to600K with global clip0.1; original milestones",
        f"X2 continuation {start} to {stop}; global clip0.1; original milestones",
    )
    assert "CLIP_LIMIT = 0.1" in source
    assert "previous.train(2)" in source
    path = folder / "continue.py"
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")
    return path

def verify_phase(folder, run, start, stop, lr, next_lr):
    checkpoint_check(run, stop, next_lr)
    history = list(records(run / "epochs.jsonl"))
    assert [row["update"] for row in history] == list(range(1000, stop + 1, 1000))
    phase = [row for row in history if start < row["update"] <= stop]
    assert len(phase) == (stop - start) // 1000
    for row in phase:
        for label in LABELS:
            assert all(math.isfinite(row["metrics"][label][key])
                       for key in ("psnr", "ssim"))

    count = clipped = 0
    max_loss = max_grad = (0.0, None)
    for row in records(folder / "clip-updates.jsonl"):
        count += 1
        assert row["update"] == start + count
        assert row["lr_used"] == [lr] and row["clip_limit"] == 0.1
        loss, grad = row["mean_l1"], row["gradient_norm_before_clip"]
        assert math.isfinite(loss) and math.isfinite(grad)
        clipped += int(row["clip_applied"])
        max_loss = max(max_loss, (loss, row["update"]))
        max_grad = max(max_grad, (grad, row["update"]))
    assert count == stop - start

    selected = min(
        phase, key=lambda row: (-row["metrics"]["DIV2K_valid"]["psnr"], row["update"])
    )
    path = run / f"epoch{selected['epoch']:05d}_u{selected['update']}.pt"
    assert path.is_file()
    baseline = history[start // 1000 - 1]
    final = phase[-1]
    return {
        "run": str(run),
        "update": stop,
        "latest_sha256": digest(run / "latest.pt"),
        "lr_for_next_update": next_lr,
        "clip_limit": 0.1,
        "updates": count,
        "clip_trigger_count": clipped,
        "max_train_l1_value_and_update": max_loss,
        "max_preclip_gradient_norm_value_and_update": max_grad,
        "selected_update": selected["update"],
        "selected_checkpoint": str(path),
        "selected_metrics": selected["metrics"],
        "fixed_metrics": final["metrics"],
        "fixed_delta_psnr_vs_phase_start": {
            label: final["metrics"][label]["psnr"] - baseline["metrics"][label]["psnr"]
            for label in LABELS
        },
        "last_10_eval_mean_psnr": {
            label: mean(row["metrics"][label]["psnr"] for row in phase[-10:])
            for label in LABELS
        },
    }

status("PREFLIGHT_X2_600K")
assert (PREVIOUS / "status.txt").read_text().strip() == "COMPLETE_X2_CLIP01_600K_NO_SHUTDOWN"
source_run = RUNS / ID600
checkpoint = source_run / "latest.pt"
assert digest(checkpoint) == SHA600, "600K断点SHA256不一致"
cfg = checkpoint_check(source_run, 600000, 2.5e-5)
assert cfg == json.loads((ROOT / "logs/multiscale50k/x2-F0.json").read_text())
assert cfg["arm"] == "F0" and cfg["model"]["upscale"] == 2
assert cfg["seed"] == 10 and cfg["self_ensemble"] is True
assert cfg["hr_patch"] == 128 and cfg["crop_border"] == 2
assert cfg["record_updates"] == 1000

jobs = (
    ("to800k", ID600, ID800, 600000, 800000, 2.5e-5, 1.25e-5),
    ("to1000k", ID800, ID1000, 800000, 1000000, 1.25e-5, 1.25e-5),
)
entries = {}
for label, source_id, target_id, start, stop, lr, _ in jobs:
    target = RUNS / target_id
    assert not target.exists() and not target.is_symlink(), target
    entries[label] = make_entry(STAGE / label, source_id, target_id, start, stop, lr)

# Estimate storage for both checkpoint series and log copies.
required = 415 * checkpoint.stat().st_size + 5 * 1024**3
free = shutil.disk_usage(RUNS).free
print(f"可用 {free / 1024**3:.2f} GiB，预计需要 {required / 1024**3:.2f} GiB")
assert free >= required, "持久化磁盘空间不足"

results = {}
for label, source_id, target_id, start, stop, lr, next_lr in jobs:
    folder = STAGE / label
    source = RUNS / source_id
    run = RUNS / target_id
    source_hash = digest(source / "latest.pt")
    expected_hash = SHA600 if start == 600000 else results["to800k"]["latest_sha256"]
    assert source_hash == expected_hash
    (folder / "provenance.json").write_text(json.dumps({
        "source_checkpoint": str(source / "latest.pt"),
        "source_sha256": source_hash,
        "start_update": start,
        "stop_update": stop,
        "lr_used": lr,
        "clip_limit": 0.1,
    }, indent=2) + "\n")

    status(f"TRAINING_X2_{start}_TO_{stop}")
    subprocess.run([sys.executable, "-u", str(entries[label])], cwd=ROOT, check=True)

    status(f"VERIFYING_X2_{stop}")
    summary = verify_phase(folder, run, start, stop, lr, next_lr)
    assert digest(source / "latest.pt") == source_hash
    results[label] = summary
    (folder / "stage-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"PASS {stop}，下一步LR={next_lr:g}，"
        f"固定权重验证PSNR={summary['fixed_metrics']['DIV2K_valid']['psnr']:.6f}，"
        f"末10次均值={summary['last_10_eval_mean_psnr']['DIV2K_valid']:.6f}",
        flush=True,
    )

selected = max(
    results.values(),
    key=lambda row: (row["selected_metrics"]["DIV2K_valid"]["psnr"],
                     -row["selected_update"]),
)
(STAGE / "result.json").write_text(json.dumps({
    "selected_update_after_600k": selected["selected_update"],
    "selected_checkpoint": selected["selected_checkpoint"],
    "selected_metrics": selected["selected_metrics"],
    "fixed_1000k_metrics": results["to1000k"]["fixed_metrics"],
}, indent=2) + "\n")
status("COMPLETE_X2_CLIP01_1000K_NO_SHUTDOWN")
