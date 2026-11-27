"""Render Figure 3 from verified original SR PNGs and reviewed ROI records."""
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


METHODS = ("bicubic", "span52_official_single", "swinir_light_single", "ngswin_single")
WIDTH_PT = 16.6 / 2.54 * 72
MARGIN, CONTEXT, GAP, COL_GAP = 4.0, 78.0, 8.0, 3.0
SIDE = (WIDTH_PT - 2 * MARGIN - CONTEXT - GAP - 5 * COL_GAP) / 6
ROW = SIDE + 32.0
HEIGHT_PT = 8 + 4 * ROW + 25


def render(records, mode, loader, basename):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"})
    figure = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72), dpi=300)
    text_limits = []

    def label(x, y_from_top, value, width, color="black", weight="normal"):
        text = figure.text(x / WIDTH_PT, 1 - y_from_top / HEIGHT_PT, value,
                           fontsize=8.5, ha="center", va="top", color=color,
                           fontweight=weight)
        text_limits.append((text, x, y_from_top, width))

    def axes(x, y_from_top, width, height):
        axis = figure.add_axes([x / WIDTH_PT, 1 - (y_from_top + height) / HEIGHT_PT,
                                width / WIDTH_PT, height / HEIGHT_PT])
        axis.axis("off")
        return axis

    for index, record in enumerate(records):
        scene = loader(record, mode)
        top = 8 + index * ROW
        overview = axes(MARGIN, top, CONTEXT, SIDE)
        overview.imshow(scene["overview"], interpolation="nearest", resample=False)
        if not scene.get("overview_already_marked", False):
            x, y, width, height = record["roi"]
            overview.add_patch(Rectangle((x - 0.5, y - 0.5), width, height,
                                         fill=False, edgecolor="#E53935", linewidth=2.0))
        center = MARGIN + CONTEXT / 2
        label(center, top + SIDE + 3, record["dataset"] + " (×4)", CONTEXT)
        label(center, top + SIDE + 14, Path(record["image"]).stem, CONTEXT)
        names = ("Bicubic", "SPAN52", "SwinIR-light", "NGswin",
                 "Ours (×8)" if mode == "ours_x8" else "Ours", "GT")
        for column, (method, name) in enumerate(zip((*METHODS, mode, "GT"), names)):
            left = MARGIN + CONTEXT + GAP + column * (SIDE + COL_GAP)
            axis = axes(left, top, SIDE, SIDE)
            axis.imshow(scene["crops"][method], interpolation="nearest", resample=False)
            center = left + SIDE / 2
            color = "#B71C1C" if method == mode else "black"
            label(center, top + SIDE + 3, name, SIDE, color,
                  "bold" if method == mode else "normal")
            if method != "GT":
                metric = record["metrics"][method]
                label(center, top + SIDE + 14,
                      f"{metric['psnr']:.2f}/{metric['ssim']:.4f}", SIDE, color)

    footer = "ROI PSNR (dB) / SSIM on Y; shave 4. SPAN52 uses released pretrained weights."
    mode_note = ("Ours: D4 ×8 self-ensemble. Other reconstructions: single pass."
                 if mode == "ours_x8" else "All reconstructions use single-pass inference.")
    figure.text(MARGIN / WIDTH_PT, 17 / HEIGHT_PT, footer, fontsize=7.5, va="top")
    figure.text(MARGIN / WIDTH_PT, 10 / HEIGHT_PT, mode_note, fontsize=7.5, va="top")

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = []
    for text, center, top, width in text_limits:
        box = text.get_window_extent(renderer)
        allowed_width = width / 72 * figure.dpi
        if box.width > allowed_width + 0.5:
            raise ValueError(f"Label wider than its column: {text.get_text()}")
        if box.x0 < 0 or box.x1 > figure.bbox.width or box.y0 < 0 or box.y1 > figure.bbox.height:
            raise ValueError(f"Clipped label: {text.get_text()}")
        if any(box.overlaps(other) for other in boxes):
            raise ValueError(f"Overlapping label: {text.get_text()}")
        boxes.append(box)
    figure.savefig(str(basename) + ".png", dpi=300)
    figure.savefig(str(basename) + ".pdf", dpi=300)
    plt.close(figure)
    with Image.open(str(basename) + ".png") as image:
        if any(abs(value - 300) > 0.1 for value in image.info["dpi"]):
            raise ValueError("PNG DPI mismatch")
    return dict(width_cm=16.6, height_cm=HEIGHT_PT / 72 * 2.54,
                dpi=300, label_font_pt=8.5, text_overlap_check="PASS",
                png=str(basename) + ".png", pdf=str(basename) + ".pdf")


def main():
    project, root, stage = map(Path, sys.argv[1:4])
    roi_folder = root / "step8-roi-review"
    if not (roi_folder / "COMPLETE").is_file():
        raise RuntimeError("ROI review has not completed")
    report = json.loads((roi_folder / "roi_ranking.json").read_text())
    expected_hashes = {(r["dataset"], r["image"]): r["sha256"] for r in report["inputs"]}
    sys.path.insert(0, str(project / "research"))
    from runtime import components
    _, metrics = components()
    output = root / "figures_paper_ready"
    output.mkdir(exist_ok=True)
    checked_metrics = []

    def loader(record, mode):
        x, y, width, height = record["roi"]
        images, crops = {}, {}
        for method in (*METHODS, mode, "GT"):
            path = root / "sr_outputs" / record["dataset"] / method / record["image"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_hashes[(record["dataset"], record["image"])][method]:
                raise ValueError(f"Original PNG changed: {path}")
            with Image.open(path) as source:
                images[method] = np.array(source.convert("RGB"))
            crops[method] = images[method][y:y + height, x:x + width]
            if crops[method].shape != (height, width, 3):
                raise ValueError(f"Invalid crop: {path}")
        for method in (*METHODS, mode):
            kw = dict(crop_border=4, test_y_channel=True)
            psnr = float(metrics.calculate_psnr(crops[method][:, :, ::-1],
                                                crops["GT"][:, :, ::-1], **kw))
            ssim = float(metrics.calculate_ssim(crops[method][:, :, ::-1],
                                                crops["GT"][:, :, ::-1], **kw))
            reference = record["metrics"][method]
            if abs(psnr - reference["psnr"]) > 1e-7 or abs(ssim - reference["ssim"]) > 1e-9:
                raise ValueError(f"ROI metric mismatch: {record['image']} {method}")
            checked_metrics.append(dict(
                figure_mode=mode, dataset=record["dataset"], image=record["image"],
                x=x, y=y, w=width, h=height, method=method, psnr_y=psnr, ssim_y=ssim))
        return dict(overview=images["GT"], crops=crops)

    figures = {}
    for mode, name in (("ours_x8", "fig3_qualitative_x4"),
                       ("ours_single", "fig3_qualitative_x4_single")):
        basename = output / name
        if basename.with_suffix(".png").exists() or basename.with_suffix(".pdf").exists():
            raise FileExistsError(f"Figure already exists: {basename}")
        figures[mode] = render(report["figure3_candidates"][mode], mode, loader, basename)
        print("FIGURE_PASS", mode, figures[mode], flush=True)
    with (stage / "fig3-roi-metrics.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checked_metrics[0]))
        writer.writeheader()
        writer.writerows(checked_metrics)
    manifest = dict(figures=figures, scenes=report["figure3_candidates"],
                    selection="Reviewed automatic local-advantage ROIs; coordinates retained",
                    comparison="Same local LR source per scene; Ours FT2 20K RAW; external released weights",
                    source_roi_report=str(roi_folder / "roi_ranking.json"))
    (stage / "fig3-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    (stage / "图3图注.txt").write_text(
        "图3：×4超分辨率局部视觉对比。红框表示放大区域，标注为该选区Y通道PSNR（dB）与SSIM，"
        "计算前从选区四边各裁去4像素。SPAN52、SwinIR-light与NGswin采用发布的预训练权重。"
        "Ours采用FT2 20K RAW权重。主版本中的Ours使用D4×8自集成，其余方法为单次推理；"
        "single版本中全部方法均为单次推理。两套图分别使用记录的自动优选ROI，放大采用最近邻插值。"
        "所展示选区用于观察局部重建差异，整库平均指标另行报告。\n", encoding="utf-8")
    with zipfile.ZipFile(stage / "fig3-paper-ready.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for record in figures.values():
            for key in ("png", "pdf"):
                path = Path(record[key])
                archive.write(path, path.name)
        for name in ("fig3-roi-metrics.csv", "fig3-manifest.json", "图3图注.txt"):
            archive.write(stage / name, name)
    (stage / "COMPLETE").write_text("Figure 3 exported; original PNG hashes and ROI metrics verified.\n")
    print("COMPLETE: FIG3_PAPER_READY", stage / "fig3-paper-ready.zip", flush=True)


if __name__ == "__main__":
    main()

