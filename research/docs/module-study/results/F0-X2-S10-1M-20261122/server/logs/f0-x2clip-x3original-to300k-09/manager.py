import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
STAGE = Path(__file__).resolve().parent
JOBS = {
    2: (
        "F0-X2-CLIP01-U210K-08",
        "F0-X2-CLIP01-U300K-09",
        210000,
        1e-4,
    ),
    3: (
        "MS-X3-F0-S10-100K-01",
        "F0-X3-ORIGINAL-U300K-09",
        100000,
        2e-4,
    ),
}
DATASETS = ("DIV2K_valid", "Set5", "Set14", "B100", "Urban100", "Manga109")

def status(text):
    (STAGE / "status.txt").write_text(text + "\n")
    print(text, flush=True)

def load_stage300():
    path = ROOT / "logs/f0-x2x3-300k-01/continue.py"
    spec = importlib.util.spec_from_file_location("stage300", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def checkpoint_check(path, update, rate):
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["update"] == update
    assert state["epoch"] == update // 1000
    assert state["scheduler"]["last_epoch"] == update
    assert state["scheduler"]["_last_lr"] == [rate]
    assert state["scheduler"]["base_lrs"] == [2e-4]
    assert dict(state["scheduler"]["milestones"]) == {
        200000: 1, 400000: 1, 600000: 1, 800000: 1
    }
    assert state["scheduler"]["gamma"] == 0.5
    assert all(
        group["lr"] == rate
        for group in state["optimizer"]["param_groups"]
    )
    for key in (
        "model", "optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"
    ):
        assert state[key] is not None
    return state["fingerprint"]

def history_check(run, end):
    rows = [
        json.loads(line)
        for line in (run / "epochs.jsonl").open()
        if line.strip()
    ]
    assert [r["update"] for r in rows] == list(range(1000, end + 1, 1000))
    count = 0
    with (run / "loss_steps.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            assert row["update"] == count
            assert math.isfinite(row["train_l1"])
            assert row["lr"] == (2e-4 if count < 200000 else 1e-4)
    assert count == end
    return rows

def main():
    status("PREFLIGHT_X2_210K_X3_100K")
    shared = load_stage300().previous
    estimate = 0

    for scale, (old_id, new_id, start, rate) in JOBS.items():
        old = ROOT / "runs" / old_id
        new = ROOT / "runs" / new_id
        assert not new.exists() and not new.is_symlink(), new
        assert not (old / "ANOMALY.json").exists(), old
        fingerprint = checkpoint_check(old / "latest.pt", start, rate)
        history_check(old, start)
        cfg = json.loads(
            (ROOT / f"logs/multiscale50k/x{scale}-F0.json").read_text()
        )
        manifest = json.loads((old / "manifest.json").read_text())
        assert manifest["config"] == cfg
        assert manifest["fingerprint"] == fingerprint
        assert cfg["seed"] == 10 and cfg["arm"] == "F0"
        assert cfg["model"]["upscale"] == scale
        assert cfg["hr_patch"] == 64 * scale and cfg["crop_border"] == scale
        assert cfg["self_ensemble"] is True
        assert cfg["selection_dataset"] == "DIV2K_valid"
        assert cfg["effective_batch"] == 64 and cfg["micro_batch"] == 16

        # Match each source to the completed preceding stage receipt.
        if scale == 2:
            receipt = json.loads(
                (ROOT / "logs/x2-clip01-200k-210k-08/stage-summary.json").read_text()
            )
            assert receipt["update"] == start and receipt["clip_limit"] == 0.1
        else:
            receipt = json.loads(
                (ROOT / "logs/f0-x2x3-100k-01/summary.json").read_text()
            )["x3"]
            assert shared.sha256(old / "latest.pt") == receipt["latest_100k_sha256"]

        size = sum(p.stat().st_size for p in old.rglob("*") if p.is_file())
        estimate += size + ((300000-start)//1000 + 5) * (
            old / "latest.pt"
        ).stat().st_size
        print(f"PASS X{scale}: start={start}, lr={rate}", flush=True)

    required = estimate + 10 * 1024**3
    free = shutil.disk_usage(ROOT).free
    print(f"Disk free={free/1024**3:.2f} GiB, required={required/1024**3:.2f} GiB")
    assert free >= required, "请扩容后再启动，空间不足"

    results = {}
    for scale, (old_id, new_id, start, rate) in JOBS.items():
        old = ROOT / "runs" / old_id
        new = ROOT / "runs" / new_id
        digest = shared.sha256(old / "latest.pt")
        status(f"TRAINING_X{scale}_{start}_TO_300000")

        if scale == 2:
            command = [sys.executable, "-u", str(STAGE / "x2.py")]
        else:
            shutil.copytree(old, new)
            assert shared.sha256(new / "latest.pt") == digest
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()), "--x3"
            ]

        (STAGE / f"x{scale}-provenance.json").write_text(json.dumps({
            "source_run": str(old),
            "destination_run": str(new),
            "start_update": start,
            "stop_update": 300000,
            "source_checkpoint_sha256": digest,
            "gradient_clip": 0.1 if scale == 2 else None,
        }, indent=2))

        subprocess.run(command, cwd=ROOT, check=True)
        checkpoint_check(new / "latest.pt", 300000, 1e-4)
        rows = history_check(new, 300000)
        assert shared.sha256(old / "latest.pt") == digest
        selected = min(
            rows,
            key=lambda row: (-row["metrics"]["DIV2K_valid"]["psnr"], row["update"])
        )
        for row in rows:
            for dataset in DATASETS:
                for label in (dataset, dataset + "_x8"):
                    assert all(
                        math.isfinite(row["metrics"][label][metric])
                        for metric in ("psnr", "ssim")
                    )
        results[f"x{scale}"] = {
            "run": str(new),
            "gradient_clip": 0.1 if scale == 2 else None,
            "selected_update": selected["update"],
            "selected_metrics": selected["metrics"],
            "fixed_300k_metrics": rows[-1]["metrics"],
            "latest_sha256": shared.sha256(new / "latest.pt"),
        }
        (STAGE / "summary.json").write_text(json.dumps(results, indent=2))

    status("COMPLETE_X2_CLIP01_X3_ORIGINAL_300K_NO_SHUTDOWN")

if __name__ == "__main__":
    if sys.argv[1:] == ["--x3"]:
        previous = load_stage300().previous
        previous.ids = lambda scale: ("unused", JOBS[3][1])
        previous.STOP = 300000
        previous.train(3)
    else:
        main()
