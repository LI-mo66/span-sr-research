import hashlib
import importlib.util
import inspect
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path.cwd()
DIAG = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/F0-X2-CLIP01-U300K-09"
RUNID = "F0-X2-CLIP01-U400K-13"
DEST = ROOT / "runs" / RUNID

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

checkpoint = SOURCE / "latest.pt"
original_hash = sha(checkpoint)
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert state["update"] == 300000
assert state["scheduler"]["last_epoch"] == 300000
assert all(g["lr"] == 1e-4 for g in state["optimizer"]["param_groups"])
assert state["scheduler"]["_last_lr"] == [1e-4]
assert state["scheduler"]["base_lrs"] == [2e-4]
assert dict(state["scheduler"]["milestones"]) == {200000: 1, 400000: 1, 600000: 1, 800000: 1}
del state

assert not DEST.exists(), DEST
required = ("latest.pt", "manifest.json", "epochs.jsonl", "loss_steps.jsonl")
size = sum((SOURCE / name).stat().st_size for name in required)
assert shutil.disk_usage(ROOT).free > size + 2 * 1024**3, "诊断磁盘空间不足"
DEST.mkdir()
for name in required:
    shutil.copy2(SOURCE / name, DEST / name)
    assert sha(SOURCE / name) == sha(DEST / name)

expected = {}

spec = importlib.util.spec_from_file_location(
    "stage300", ROOT / "logs/f0-x2x3-300k-01/continue.py"
)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
previous = stage.previous
previous.ids = lambda scale: ("unused", RUNID)
assert previous.NEW_CAP.count("300000") == 1
previous.NEW_CAP = previous.NEW_CAP.replace("300000", "400000")
previous.STOP = 400000

sys.path[:0] = [
    str(ROOT / "research/multiscale_f0"),
    str(ROOT / "research"),
]
import fusion
import bound

stream = (DIAG / "observations.jsonl").open("w", buffering=1)
current = {"update": 199001, "micro": 0}
activations = {}
bound_stats = []
matches = []
losses = []
trigger = None

def emit(row):
    stream.write(json.dumps(row, allow_nan=False) + "\n")

def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)

original_bound = bound.bounded

def observed_bound(x, *args, **kwargs):
    y = original_bound(x, *args, **kwargs)
    if current["update"] >= 199440:
        knee = kwargs.get("knee", args[0] if args else 8.0)
        with torch.no_grad():
            bound_stats.append({
                "input_absmax": finite_number(x.detach().abs().max()),
                "output_absmax": finite_number(y.detach().abs().max()),
                "above_knee_fraction": float((x.detach().abs() > knee).float().mean()),
            })
    return y

# Replace references already imported by candidate modules.
for module in list(sys.modules.values()):
    namespace = getattr(module, "__dict__", None)
    if namespace is None:
        continue
    for name, value in list(namespace.items()):
        if value is original_bound:
            namespace[name] = observed_bound

original_build = fusion.build_model

def wrapped_build(*args, **kwargs):
    model = original_build(*args, **kwargs)
    def hook(name):
        def record(module, inputs, output):
            if current["update"] < 199440:
                return
            values = output if isinstance(output, (tuple, list)) else (output,)
            stats = []
            with torch.no_grad():
                for value in values:
                    if torch.is_tensor(value) and value.is_floating_point():
                        stats.append({
                            "shape": list(value.shape),
                            "finite": bool(torch.isfinite(value).all()),
                            "absmax": finite_number(value.detach().abs().max()),
                            "mean": finite_number(value.detach().mean()),
                        })
            if stats:
                activations[name] = stats
        return record
    for name, module in model.named_modules():
        # Inspect block and adapter boundaries plus fusion/reconstruction outputs.
        if name and (
            "." not in name
            or name.endswith((".original", ".adapter", ".spatial_gate"))
        ):
            module.register_forward_hook(hook(name))
    return model

fusion.build_model = wrapped_build
original_l1 = F.l1_loss

def observed_loss(prediction, target, *args, **kwargs):
    global trigger
    result = original_l1(prediction, target, *args, **kwargs)
    frame = inspect.currentframe().f_back
    local = frame.f_locals
    if "update" not in local or "opt" not in local or "model" not in local:
        return result
    update = local["update"] + 1
    micro = current["micro"]
    raw = float((prediction.detach() - target.detach()).abs().mean())
    losses.append(raw)
    if update >= 199440:
        with torch.no_grad():
            per_image = (prediction.detach() - target.detach()).abs().flatten(1).mean(1)
        emit({
            "kind": "microbatch", "update": update, "micro": micro,
            "l1": finite_number(raw),
            "per_image_l1": [finite_number(v) for v in per_image],
            "activations": dict(activations),
            "bound_calls": list(bound_stats),
        })
        print("DIAG", update, micro, "L1", raw, flush=True)

    if update == 199450 and micro == 0 or not math.isfinite(raw) or raw > 1.0:
        label = "before-onset" if update == 199450 and micro == 0 else "trigger"
        torch.save({
            "update": update, "micro": micro,
            "lr": local["lr"].detach().cpu(),
            "hr": target.detach().cpu(),
            "prediction": prediction.detach().cpu(),
            "model": local["model"].state_dict(),
            "optimizer": local["opt"].state_dict(),
            "scheduler": local["sched"].state_dict(),
        }, DIAG / f"{label}-state.pt")
        if label == "trigger":
            trigger = {"update": update, "micro": micro, "l1": finite_number(raw)}
    if micro == 3:
        measured = sum(losses) / 4
        if update in expected:
            difference = abs(measured - expected[update])
            matches.append(difference)
            if False:
                raise RuntimeError(
                    f"重放不一致 update={update}, abs_difference={difference}"
                )
        losses.clear()
        current["micro"] = 0
        current["update"] = update + 1
    else:
        current["micro"] += 1
    activations.clear()
    bound_stats.clear()
    return result


CLIP_LIMIT = 0.1
adam_original = torch.optim.Adam.step
clip_stream = (DIAG / "clip-updates.jsonl").open("w", buffering=1)

def clipped_adam_step(optimizer, *args, **kwargs):
    frame = inspect.currentframe().f_back
    local = None
    while frame is not None:
        candidate = frame.f_locals
        if candidate.get("opt") is optimizer and "model" in candidate and "update" in candidate:
            local = candidate
            break
        frame = frame.f_back
    if local is None:
        raise RuntimeError("Cannot locate trainer frame for clipping")

    update = local["update"] + 1
    parameters = [
        (name, parameter)
        for name, parameter in local["model"].named_parameters()
        if parameter.requires_grad
    ]
    snapshots = [parameter.detach().clone() for _, parameter in parameters]
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in parameters],
        CLIP_LIMIT,
        error_if_nonfinite=True,
    )
    result = adam_original(optimizer, *args, **kwargs)
    changes = []
    with torch.no_grad():
        for (name, parameter), before in zip(parameters, snapshots):
            changes.append({
                "name": name,
                "update_norm": float((parameter.detach() - before).norm()),
            })
    row = {
        "update": update,
        "clip_limit": CLIP_LIMIT,
        "gradient_norm_before_clip": float(norm),
        "clip_applied": float(norm) > CLIP_LIMIT,
        "mean_l1": local.get("loss_value"),
        "lr_used": [group["lr"] for group in optimizer.param_groups],
        "largest_updates": sorted(
            changes, key=lambda item: item["update_norm"], reverse=True
        )[:5],
    }
    clip_stream.write(json.dumps(row) + "\n")
    if update % 100 == 0 or 199445 <= update <= 199460:
        print("CLIP", json.dumps(row), flush=True)
    return result

torch.optim.Adam.step = clipped_adam_step

F.l1_loss = original_l1

report = {"source_checkpoint_sha256": original_hash, "clip_limit": 0.1, "scope": "X2 continuation300K to400K with global clip0.1; original milestones"}
try:
    previous.train(2)
    report["status"] = "REPLAY_FINISHED_WITHOUT_GUARD_TRIGGER"
except Exception as error:
    report.update(
        status="REPRODUCED_ANOMALY" if trigger else "STOPPED_READ_ERROR",
        error=f"{type(error).__name__}: {error}",
    )
    raise
finally:
    report.update(
        trigger=trigger,
        compared_updates=len(matches),
        maximum_loss_replay_difference=max(matches) if matches else None,
        source_checkpoint_unchanged=sha(checkpoint) == original_hash,
        before_onset_saved=(DIAG / "before-onset-state.pt").is_file(),
        trigger_saved=(DIAG / "trigger-state.pt").is_file(),
    )
    (DIAG / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    stream.close()
    print(json.dumps(report, indent=2), flush=True)
