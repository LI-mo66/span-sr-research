import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
STAGE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "logs/f0-x2x3-100k-01"
START, STOP = 100000, 300000

spec = importlib.util.spec_from_file_location(
    "verified_previous_stage", PREVIOUS / "continue.py"
)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

# Reuse the audited trainer and change only the authorized execution cap.
assert previous.START == 50000 and previous.STOP == 100000
assert previous.NEW_CAP.count("100000") == 1
previous.NEW_CAP = previous.NEW_CAP.replace("100000", "300000")
previous.STOP = STOP
previous.ids = lambda scale: (
    f"MS-X{scale}-F0-S10-100K-01",
    f"MS-X{scale}-F0-S10-300K-01",
)


def status(message):
    temporary = STAGE / "status.tmp"
    temporary.write_text(message + "\n", encoding="utf-8")
    temporary.replace(STAGE / "status.txt")
    print(message, flush=True)


def check_checkpoint(path, fingerprint, update):
    state = previous.torch.load(
        path, map_location="cpu", weights_only=False
    )
    assert state["fingerprint"] == fingerprint, path
    assert state["update"] == update, path
    assert state["epoch"] == update // 1000, path
    assert state["epoch_record"]["update"] == update, path
    scheduler = state["scheduler"]
    assert scheduler["last_epoch"] == update, path
    assert dict(scheduler["milestones"]) == dict.fromkeys(
        [200000, 400000, 600000, 800000], 1
    ), path
    assert scheduler["gamma"] == 0.5, path
    groups = state["optimizer"]["param_groups"]
    rate = 2e-4 if update < 200000 else 1e-4
    assert groups and state["optimizer"]["state"], path
    assert scheduler["base_lrs"] == [2e-4] * len(groups), path
    assert scheduler["_last_lr"] == [rate] * len(groups), path
    assert all(group["lr"] == rate for group in groups), path
    for key in (
        "model", "best", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"
    ):
        assert state[key] is not None, (path, key)


def check_history(run, end):
    history = previous.read_rows(run / "epochs.jsonl")
    assert [row["update"] for row in history] == list(
        range(1000, end + 1, 1000)
    ), run

    count = 0
    with (run / "loss_steps.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            assert row["update"] == count, (run, count)
            expected = 2e-4 if count < 200000 else 1e-4
            assert row["lr"] == expected, (run, count, row["lr"])
            assert math.isfinite(row["train_l1"]), (run, count)
    assert count == end, (run, count, end)

    for row in history:
        for dataset in previous.DATASETS:
            for label in (dataset, dataset + "_x8"):
                assert all(
                    math.isfinite(row["metrics"][label][metric])
                    for metric in ("psnr", "ssim")
                ), (run, row["update"], label)
    return history


def preflight(scale, summary):
    old_id, new_id = previous.ids(scale)
    old = ROOT / "runs" / old_id
    new = ROOT / "runs" / new_id
    if new.exists() or new.is_symlink():
        raise FileExistsError(new)

    cfg = previous.read_json(
        ROOT / "logs/multiscale50k" / f"x{scale}-F0.json"
    )
    manifest = previous.read_json(old / "manifest.json")
    assert manifest["config"] == cfg
    assert cfg["arm"] == "F0" and cfg["seed"] == 10
    assert cfg["model"]["upscale"] == scale
    assert cfg["hr_patch"] == 64 * scale and cfg["crop_border"] == scale
    assert cfg["effective_batch"] == 64 and cfg["micro_batch"] == 16
    assert cfg["record_updates"] == 1000 and cfg["loader_workers"] == 0
    assert cfg["lr"] == 2e-4 and cfg["total_updates"] == 50000
    assert cfg["milestones"] == [200000, 400000, 600000, 800000]
    assert cfg["gamma"] == 0.5 and cfg["self_ensemble"] is True
    assert cfg["selection_dataset"] == "DIV2K_valid"

    receipt = summary[f"x{scale}"]
    assert Path(receipt["run"]).resolve() == old.resolve()
    digest = previous.sha256(old / "latest.pt")
    assert digest == receipt["latest_100k_sha256"], old
    assert not (old / "ANOMALY.json").exists(), old
    assert not any(path.is_symlink() for path in old.rglob("*")), old

    check_checkpoint(old / "latest.pt", manifest["fingerprint"], START)
    check_history(old, START)
    print(f"PASS X{scale}: 100K checkpoint SHA256, state and history", flush=True)
    return scale, old, new, manifest["fingerprint"], digest


def main():
    status("PREFLIGHT_BOTH_100K_CHECKPOINTS")
    previous.compiled_trainer()
    summary_100k = previous.read_json(PREVIOUS / "summary.json")
    jobs = [preflight(scale, summary_100k) for scale in (2, 3)]

    estimate = sum(
        sum(p.stat().st_size for p in old.rglob("*") if p.is_file())
        + 210 * (old / "latest.pt").stat().st_size
        for _, old, _, _, _ in jobs
    )
    available = shutil.disk_usage(ROOT).free
    print(
        f"Disk free={available / 1024**3:.2f} GiB; "
        f"estimated requirement with margin={(estimate + 10 * 1024**3) / 1024**3:.2f} GiB",
        flush=True,
    )
    if available < estimate + 10 * 1024**3:
        raise RuntimeError("Insufficient disk for both 300K continuation runs")

    results = {}
    for scale, old, new, fingerprint, digest in jobs:
        status(f"COPYING_X{scale}_100K_TO_INDEPENDENT_RUN")
        shutil.copytree(old, new)
        assert previous.sha256(new / "latest.pt") == digest
        provenance = dict(
            source_run=str(old),
            continuation_run=str(new),
            checkpoint_100k_sha256=digest,
            start_update=START,
            stop_update=STOP,
            seed=10,
            self_ensemble=True,
            selection_dataset="DIV2K_valid",
            fingerprint=fingerprint,
            runtime_cap=STOP,
        )
        (STAGE / f"x{scale}-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

        status(f"TRAINING_X{scale}_F0_100000_TO_300000")
        subprocess.run(
            [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--train", str(scale),
            ],
            cwd=ROOT,
            check=True,
        )

        status(f"VERIFYING_X{scale}_300K")
        check_checkpoint(new / "latest.pt", fingerprint, STOP)
        check_checkpoint(
            new / "epoch00300_u300000.pt", fingerprint, STOP
        )
        history = check_history(new, STOP)
        assert previous.sha256(old / "latest.pt") == digest
        select = lambda row: (
            -row["metrics"]["DIV2K_valid"]["psnr"], row["update"]
        )
        old_best = min(
            (row for row in history if row["update"] <= START),
            key=select,
        )
        selected = min(history, key=select)
        selected_path = new / (
            f"epoch{selected['epoch']:05d}_u{selected['update']}.pt"
        )
        assert selected_path.is_file()
        results[f"x{scale}"] = dict(
            run=str(new),
            latest_300k_sha256=previous.sha256(new / "latest.pt"),
            selected_0_to_100k=old_best["update"],
            selected_0_to_300k=selected["update"],
            selected_checkpoint=str(selected_path),
            selected_metrics=selected["metrics"],
            fixed_300k_metrics=history[-1]["metrics"],
            delta_selected_psnr_vs_100k={
                name: value["psnr"] - old_best["metrics"][name]["psnr"]
                for name, value in selected["metrics"].items()
            },
        )
        (STAGE / "summary.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )

    status("COMPLETE_X2_X3_F0_300K_NO_SHUTDOWN")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--train":
        scale = int(sys.argv[2])
        assert scale in (2, 3)
        previous.train(scale)
    else:
        main()
