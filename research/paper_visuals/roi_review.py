"""Search recorded SR PNGs and export traceable qualitative-review candidates."""
import csv
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


METHODS = ("bicubic", "span52_official_single", "swinir_light_single",
           "ngswin_single", "ours_single", "ours_x8", "GT")
OTHERS = METHODS[:4]
SIZE, STRIDE, EDGE, SHAVE = 96, 16, 16, 4


def read_images(root, dataset, name):
    images, hashes = {}, {}
    for method in METHODS:
        path = root / "sr_outputs" / dataset / method / name
        raw = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.dtype != np.uint8:
            raise ValueError(f"Invalid PNG: {path}")
        images[method] = image
        hashes[method] = hashlib.sha256(raw).hexdigest()
    if len({image.shape for image in images.values()}) != 1:
        raise ValueError(f"Mismatched image shapes: {dataset}/{name}")
    return images, hashes


def mean_rectangles(image, x, y, width):
    integral = cv2.integral(image.astype(np.float64), sdepth=cv2.CV_64F)
    return (integral[y + width, x + width] - integral[y, x + width]
            - integral[y + width, x] + integral[y, x]) / width**2


def overlap(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
                or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def search(images, metrics):
    height, width = images["GT"].shape[:2]
    xx, yy = np.meshgrid(np.arange(EDGE, width - EDGE - SIZE + 1, STRIDE),
                         np.arange(EDGE, height - EDGE - SIZE + 1, STRIDE))
    x, y = xx.ravel(), yy.ravel()
    if not len(x):
        raise ValueError(f"Image too small for fixed ROI: {height}x{width}")
    gt_y = metrics.to_y_channel(images["GT"]).squeeze(-1).astype(np.float64)
    dx = cv2.Sobel(gt_y, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gt_y, cv2.CV_64F, 0, 1, ksize=3)
    texture = mean_rectangles(cv2.magnitude(dx, dy), x, y, SIZE)
    retained = texture >= np.quantile(texture, 0.7)
    scores = {}
    for method in METHODS[:-1]:
        error = metrics.to_y_channel(images[method]).squeeze(-1).astype(np.float64) - gt_y
        mse = mean_rectangles(error**2, x + SHAVE, y + SHAVE, SIZE - 2 * SHAVE)
        scores[method] = 10 * np.log10(255.0**2 / np.maximum(mse, 1e-12))
    strongest = np.argmax(np.stack([scores[m] for m in OTHERS]), axis=0)
    best_other = np.max(np.stack([scores[m] for m in OTHERS]), axis=0)
    winners = {}
    for mode in ("ours_single", "ours_x8"):
        margin = scores[mode] - best_other
        order = sorted(np.flatnonzero(retained),
                       key=lambda i: (-margin[i], -texture[i], int(y[i]), int(x[i])))
        chosen = []
        for i in order:
            roi = (int(x[i]), int(y[i]), SIZE, SIZE)
            if not any(overlap(roi, item["roi"]) for item in chosen):
                local_metrics = {}
                gt = images["GT"][y[i]:y[i] + SIZE, x[i]:x[i] + SIZE]
                for method in METHODS[:-1]:
                    crop = images[method][y[i]:y[i] + SIZE, x[i]:x[i] + SIZE]
                    measured = float(metrics.calculate_psnr(crop, gt, crop_border=SHAVE,
                                                            test_y_channel=True))
                    if not np.isfinite(measured) or abs(measured - scores[method][i]) > 1e-4:
                        raise ValueError("Integral/direct ROI PSNR mismatch")
                    local_metrics[method] = dict(
                        psnr=measured,
                        ssim=float(metrics.calculate_ssim(crop, gt, crop_border=SHAVE,
                                                          test_y_channel=True)))
                chosen.append(dict(
                    roi=list(roi), margin=float(margin[i]), texture=float(texture[i]),
                    strongest_other=OTHERS[int(strongest[i])], metrics=local_metrics))
            if len(chosen) == 2:
                break
        winners[mode] = chosen
    audit = (x, y, texture, retained, scores, strongest, best_other)
    return winners, audit


def figure_scenes(ranking):
    chosen = []
    for dataset, limit in (("Urban100", 2), ("Manga109", 1)):
        seen = set()
        for item in ranking:
            if item["dataset"] == dataset and item["image"] not in seen:
                chosen.append(item)
                seen.add(item["image"])
                if len(seen) == limit:
                    break
    ppt = next((r for r in ranking
                if r["dataset"] == "Set14" and Path(r["image"]).stem == "ppt3"), None)
    if ppt is not None and ppt["margin"] > 0:
        chosen.append(ppt)
    else:
        seen = {(r["dataset"], r["image"]) for r in chosen}
        extra = next((r for r in ranking if r["dataset"] == "Urban100"
                      and (r["dataset"], r["image"]) not in seen), None)
        if extra is not None:
            chosen.append(extra)
    if len(chosen) != 4:
        raise ValueError("Not enough distinct scenes")
    return chosen


def preview(root, path, record, mode):
    images, _ = read_images(root, record["dataset"], record["image"])
    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    panel, gap, context = 224, 12, 300
    canvas = Image.new("RGB", (context + 6 * (panel + gap) + 32, 360), "white")
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = record["roi"]
    title = (f"{record['dataset']} / {record['image']} | {mode} | "
             f"ROI ({x},{y},{w},{h}) | margin {record['margin']:+.3f} dB")
    draw.text((16, 10), title, font=font, fill="black")
    gt = Image.fromarray(images["GT"][:, :, ::-1])
    scale = min(context / gt.width, panel / gt.height)
    view = gt.resize((round(gt.width * scale), round(gt.height * scale)),
                     Image.Resampling.LANCZOS)
    left, top = 16 + (context - view.width) // 2, 48 + (panel - view.height) // 2
    canvas.paste(view, (left, top))
    draw.rectangle((left + x * view.width / gt.width,
                    top + y * view.height / gt.height,
                    left + (x + w) * view.width / gt.width,
                    top + (y + h) * view.height / gt.height), outline="#E53935", width=3)
    draw.text((16, 282), "HR overview / red ROI", font=small, fill="black")
    labels = ("Bicubic", "SPAN52 (official)", "SwinIR-light", "NGswin",
              "Ours (single)" if mode == "ours_single" else "Ours (x8)", "GT")
    for index, (method, label) in enumerate(zip((*OTHERS, mode, "GT"), labels)):
        left = 16 + context + gap + index * (panel + gap)
        crop = images[method][y:y + h, x:x + w, ::-1]
        patch = Image.fromarray(crop).resize((panel, panel), Image.Resampling.NEAREST)
        canvas.paste(patch, (left, 48))
        color = "#B71C1C" if method == mode else "black"
        draw.text((left, 280), label, font=font, fill=color)
        if method != "GT":
            score = record["metrics"][method]
            text = f"{score['psnr']:.2f} dB / {score['ssim']:.4f}"
            draw.text((left, 305), text, font=small, fill=color)
    draw.text((16, 337), "ROI metrics: Y channel, shave 4. Enlarged crops: NEAREST.",
              font=small, fill="black")
    canvas.save(path, dpi=(300, 300))


def main():
    project, root, stage = map(Path, sys.argv[1:4])
    sys.path.insert(0, str(project / "research"))
    from runtime import components
    _, metrics = components()
    ranking = {mode: [] for mode in ("ours_single", "ours_x8")}
    inputs = []
    with gzip.open(stage / "all_roi_scores.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset", "image", "x", "y", "w", "h", "texture",
                         "mode", "margin", "strongest_other", *METHODS[:-1]))
        for dataset, count in (("Set14", 14), ("Urban100", 100), ("Manga109", 109)):
            files = sorted((root / "sr_outputs" / dataset / "GT").glob("*.png"))
            if len(files) != count:
                raise ValueError(f"{dataset}: expected {count} GT PNGs, found {len(files)}")
            for index, path in enumerate(files, 1):
                images, hashes = read_images(root, dataset, path.name)
                selected, audit = search(images, metrics)
                inputs.append(dict(dataset=dataset, image=path.name, sha256=hashes))
                x, y, texture, retained, scores, strongest, best_other = audit
                for mode in ranking:
                    for record in selected[mode]:
                        ranking[mode].append(dict(dataset=dataset, image=path.name, **record))
                    for i in np.flatnonzero(retained):
                        writer.writerow((dataset, path.name, int(x[i]), int(y[i]), SIZE, SIZE,
                                         texture[i], mode, scores[mode][i] - best_other[i],
                                         OTHERS[int(strongest[i])],
                                         *(scores[m][i] for m in METHODS[:-1])))
                print(f"ROI_PASS {dataset} {index}/{count}", flush=True)
    selections = {}
    preview_folder = root / "candidates_preview" / stage.name
    preview_folder.mkdir(parents=True)
    for mode, records in ranking.items():
        records.sort(key=lambda r: (-r["margin"], r["dataset"], r["image"], r["roi"]))
        selections[mode] = figure_scenes(records)
        to_draw = records[:8] + selections[mode]
        drawn = set()
        for record in to_draw:
            identity = (record["dataset"], record["image"], tuple(record["roi"]))
            if identity in drawn:
                continue
            drawn.add(identity)
            x, y, _, _ = record["roi"]
            name = f"{mode}_{record['dataset']}_{Path(record['image']).stem}_{x}_{y}.png"
            preview(root, preview_folder / name, record, mode)
            record["preview"] = str((preview_folder / name).relative_to(root))
        print("FIG3_CANDIDATES", mode, flush=True)
        for record in selections[mode]:
            print(record["dataset"], record["image"], "ROI", record["roi"],
                  "margin", f"{record['margin']:+.4f}", record["preview"], flush=True)
    report = dict(
        scope="Exploratory ROI selection from measured outputs; not whole-dataset superiority",
        roi_size=SIZE, stride=STRIDE, image_edge=EDGE, roi_metric_shave=SHAVE,
        texture_rule="GT Sobel mean at or above image-specific 70th percentile",
        ranking_mse_floor=1e-12,
        competitors=list(OTHERS), rankings=ranking, figure3_candidates=selections,
        inputs=inputs,
    )
    report_path = stage / "roi_ranking.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    with zipfile.ZipFile(stage / "roi-review.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, "roi_ranking.json")
        archive.write(stage / "all_roi_scores.csv.gz", "all_roi_scores.csv.gz")
        for path in sorted(preview_folder.glob("*.png")):
            archive.write(path, "candidates_preview/" + path.name)
    (stage / "COMPLETE").write_text("ROI search and review previews completed.\n")
    print("REVIEW_ZIP", stage / "roi-review.zip", flush=True)
    print("COMPLETE: ROI_REVIEW_READY", flush=True)


if __name__ == "__main__":
    main()

