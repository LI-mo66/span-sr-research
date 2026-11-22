import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
STAGE = Path(__file__).resolve().parent
LOGS = ROOT / "logs/multiscale50k"
SOURCE = ROOT / "research/multiscale_f0/trainer.py"
RECORDS = Path("/home/featurize/work/span-x4-research/records")
START, STOP, INTERVAL = 50000, 100000, 1000
MILESTONES = [200000, 400000, 600000, 800000]
DATASETS = ("DIV2K_valid", "Set5", "Set14", "B100", "Urban100", "Manga109")

OLD_CAP = "not 0 < stop <= cfg['total_updates']"
NEW_CAP = (
    "not 0 < stop <= (100000 if args.resume and cfg['arm'] == 'F0' "
    "and SCALE in (2, 3) and cfg['total_updates'] == 50000 "
    "else cfg['total_updates'])"
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids(scale):
    return f"MS-X{scale}-F0-S10-50K-01", f"MS-X{scale}-F0-S10-100K-01"


def status(message):
    tmp = STAGE / "status.tmp"
    tmp.write_text(message + "\n", encoding="utf-8")
    tmp.replace(STAGE / "status.txt")
    print(message, flush=True)


def compiled_trainer():
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(OLD_CAP) != 1:
        raise RuntimeError("Trainer cap line differs from audited version")
    return compile(source.replace(OLD_CAP, NEW_CAP), str(SOURCE), "exec")


def check_checkpoint(path, fingerprint, update):
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["fingerprint"] == fingerprint, path
    assert state["update"] == update and state["epoch"] == update // INTERVAL, path
    assert state["epoch_record"]["update"] == update, path
    scheduler = state["scheduler"]
    assert scheduler["last_epoch"] == update, path
    assert dict(scheduler["milestones"]) == dict.fromkeys(MILESTONES, 1), path
    assert scheduler["gamma"] == 0.5, path
    groups = state["optimizer"]["param_groups"]
    assert groups and state["optimizer"]["state"], path
    assert scheduler["base_lrs"] == scheduler["_last_lr"] == [2e-4] * len(groups), path
    assert all(group["lr"] == 2e-4 for group in groups), path
    for key in ("model", "best", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"):
        assert state[key] is not None, (path, key)


def check_history(run, end):
    epochs = read_rows(run / "epochs.jsonl")
    assert [r["update"] for r in epochs] == list(range(INTERVAL, end + 1, INTERVAL)), run
    steps = read_rows(run / "loss_steps.jsonl")
    assert [r["update"] for r in steps] == list(range(1, end + 1)), run
    assert all(r["lr"] == 2e-4 and math.isfinite(r["train_l1"]) for r in steps), run
    for row in epochs:
        for dataset in DATASETS:
            for label in (dataset, dataset + "_x8"):
                assert all(
                    math.isfinite(row["metrics"][label][key])
                    for key in ("psnr", "ssim")
                ), (run, row["update"], label)
    return epochs


def preflight(scale):
    old_id, new_id = ids(scale)
    old = ROOT / "runs" / old_id
    new = ROOT / "runs" / new_id
    if new.exists() or new.is_symlink():
        raise FileExistsError(new)

    config = LOGS / f"x{scale}-F0.json"
    cfg = read_json(config)
    manifest = read_json(old / "manifest.json")
    assert manifest["id"] == old_id and manifest["config"] == cfg
    assert manifest["commit"] == "0748bd60c1eaddda4e0db7f01ba768a0f2a22be3"
    assert cfg["arm"] == "F0" and cfg["seed"] == 10
    assert cfg["model"]["upscale"] == scale and cfg["hr_patch"] == scale * 64
    assert cfg["crop_border"] == scale and cfg["self_ensemble"] is True
    assert cfg["selection_dataset"] == "DIV2K_valid"
    assert cfg["total_updates"] == START and cfg["record_updates"] == INTERVAL
    assert cfg["effective_batch"] == 64 and cfg["micro_batch"] == 16
    assert cfg["lr"] == 2e-4 and cfg["milestones"] == MILESTONES and cfg["gamma"] == 0.5
    assert cfg["loader_workers"] == 0

    receipt = LOGS / f"{old_id}-backup-path.txt"
    backup = Path(receipt.read_text(encoding="utf-8").strip()).resolve()
    assert backup.is_relative_to(RECORDS.resolve())
    assert (backup / "BACKUP_COMPLETE").is_file()
    backup_manifest = read_json(backup / "backup-manifest.json")
    assert backup_manifest["runid"] == old_id
    hashes = {r["path"]: r["sha256"] for r in backup_manifest["files"]}

    for name in ("latest.pt", "manifest.json", "epochs.jsonl", "loss_steps.jsonl"):
        expected = hashes[f"run/{name}"]
        assert sha256(old / name) == expected, (old, name)
        assert sha256(backup / "run" / name) == expected, (backup, name)

    assert not any(p.is_symlink() for p in old.rglob("*")), old
    check_checkpoint(old / "latest.pt", manifest["fingerprint"], START)
    check_history(old, START)
    print(f"PASS ×{scale}: config, backup SHA256, 50K history and full Adam checkpoint")
    return old, new, config, manifest["fingerprint"], hashes["run/latest.pt"]


def train(scale):
    _, new_id = ids(scale)
    config = LOGS / f"x{scale}-F0.json"
    code = compiled_trainer()
    sys.path[:0] = [str(SOURCE.parent), str(ROOT / "research")]
    sys.argv = [
        str(SOURCE), "--config", str(config), "--id", new_id,
        "--resume", "--until-update", str(STOP),
    ]
    exec(code, {"__name__": "__main__", "__file__": str(SOURCE)})


def main():
    status("PREFLIGHT_BOTH_F0_CHECKPOINTS")
    compiled_trainer()
    jobs = [(scale, *preflight(scale)) for scale in (2, 3)]
    estimate = sum(
        sum(p.stat().st_size for p in old.rglob("*") if p.is_file())
        + 54 * (old / "latest.pt").stat().st_size
        for _, old, _, _, _, _ in jobs
    )
    if shutil.disk_usage(ROOT).free < estimate + 4 * 1024**3:
        raise RuntimeError("Insufficient workspace disk for both continuation runs")

    summary = {}
    for scale, old, new, config, fingerprint, start_hash in jobs:
        status(f"COPYING_X{scale}_50K_TO_INDEPENDENT_RUN")
        shutil.copytree(old, new)
        assert sha256(new / "latest.pt") == start_hash
        provenance = dict(
            scale=scale, seed=10, start_update=START, stop_update=STOP,
            source_run=str(old), continuation_run=str(new),
            config=str(config), checkpoint_50k_sha256=start_hash,
            fingerprint=fingerprint, self_ensemble=True,
        )
        (STAGE / f"x{scale}-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

        status(f"TRAINING_X{scale}_F0_50000_TO_100000")
        subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--train", str(scale)],
            cwd=ROOT, check=True,
        )

        status(f"VERIFYING_X{scale}_100K")
        check_checkpoint(new / "latest.pt", fingerprint, STOP)
        rows = check_history(new, STOP)
        final = new / "epoch00100_u100000.pt"
        check_checkpoint(final, fingerprint, STOP)
        assert sha256(old / "latest.pt") == start_hash
        select = lambda row: (-row["metrics"]["DIV2K_valid"]["psnr"], row["update"])
        best_50k = min(rows[:START // INTERVAL], key=select)
        best_100k = min(rows, key=select)
        summary[f"x{scale}"] = dict(
            run=str(new), latest_100k_sha256=sha256(new / "latest.pt"),
            selected_0_to_50k=best_50k["update"],
            selected_0_to_100k=best_100k["update"],
            selected_checkpoint=str(
                new / f"epoch{best_100k['epoch']:05d}_u{best_100k['update']}.pt"
            ),
            selected_metrics=best_100k["metrics"],
            fixed_100k_metrics=rows[-1]["metrics"],
            delta_selected_psnr={
                name: score["psnr"] - best_50k["metrics"][name]["psnr"]
                for name, score in best_100k["metrics"].items()
            },
        )
        assert Path(summary[f"x{scale}"]["selected_checkpoint"]).is_file()
        (STAGE / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    status("COMPLETE_X2_X3_F0_100K_NO_SHUTDOWN")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--train":
        train(int(sys.argv[2]))
    else:
        main()
