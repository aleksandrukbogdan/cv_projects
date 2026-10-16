"""Прогон SR-inpainting: три кадра, три деградации, три маски, три метрики."""

import argparse
import csv
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image, ImageDraw, ImageFont

from degrade import HR_SIZE, make_degradations, make_masks, prepare_hr
from metrics_eval import load_metrics, region_psnr, score_pair
from sampler import super_resolve_masked


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
SOURCES = {
    "cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1280px-Cat03.jpg",
    "bridge": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Tower_Bridge_from_Shad_Thames.jpg/1280px-Tower_Bridge_from_Shad_Thames.jpg",
    "mountain": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/1280px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg",
}
MASK_ORDER = ("zeros", "combined", "ones")
DEG_ORDER = ("bicubic", "unknown", "realesrgan")
MASK_TITLE = {
    "zeros": "маска 0",
    "combined": "маска слева",
    "ones": "маска 1",
}
DEG_TITLE = {
    "bicubic": "bicubic ×4",
    "unknown": "unknown",
    "realesrgan": "RealESRGAN",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--noise-level", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("Нужна видеокарта: на CPU этот прогон слишком долгий.")

    images = load_images()
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.vae.to(dtype=torch.float32)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    metrics = load_metrics(device)

    out = args.out
    for folder in ("hr", "lq", "outputs", "grids"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    rows = []
    for name, hr in images.items():
        hr.save(out / "hr" / f"{name}.png")
        degraded = make_degradations(hr, seed=args.seed)
        for deg_name in DEG_ORDER:
            lq = degraded[deg_name]
            lq.save(out / "lq" / f"{name}_{deg_name}.png")
            masks = make_masks(lq.height, lq.width)
            outputs = {}
            for mask_name in MASK_ORDER:
                print(f"{name} / {deg_name} / {mask_name}", flush=True)
                prediction = super_resolve_masked(
                    pipe,
                    lq,
                    masks[mask_name],
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    noise_level=args.noise_level,
                    seed=args.seed,
                )
                if prediction.size != (HR_SIZE, HR_SIZE):
                    prediction = prediction.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
                prediction.save(out / "outputs" / f"{name}_{deg_name}_{mask_name}.png")
                outputs[mask_name] = prediction
                scores = score_pair(metrics, prediction, hr, device)
                left, right = _half_masks(HR_SIZE)
                scores["psnr_left"] = region_psnr(prediction, hr, left)
                scores["psnr_right"] = region_psnr(prediction, hr, right)
                row = {"image": name, "degradation": deg_name, "mask": mask_name, **scores}
                rows.append(row)
                print(
                    f"  PSNR {scores['psnr']:.2f}  LPIPS {scores['lpips']:.3f}  CLIP-IQA {scores['clipiqa']:.3f}",
                    flush=True,
                )
            bicubic = lq.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
            base = score_pair(metrics, bicubic, hr, device)
            left, right = _half_masks(HR_SIZE)
            base["psnr_left"] = region_psnr(bicubic, hr, left)
            base["psnr_right"] = region_psnr(bicubic, hr, right)
            rows.append({"image": name, "degradation": deg_name, "mask": "bicubic", **base})
            print(
                f"  bicubic  PSNR {base['psnr']:.2f}  LPIPS {base['lpips']:.3f}  CLIP-IQA {base['clipiqa']:.3f}",
                flush=True,
            )
            sheet = _sheet(bicubic, masks["combined"], outputs, hr)
            sheet.save(out / "grids" / f"{name}_{deg_name}.jpg", quality=90)

    _write_csv(out / "metrics.csv", rows)
    summary = _summarize(rows)
    _write_csv(out / "summary.csv", summary)
    _chart(summary, out / "chart.png")
    print(f"Готово: {out}")


def load_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, url in SOURCES.items():
        path = IMAGE_DIR / f"{name}.jpg"
        if not path.exists():
            print(f"Скачиваю {name}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "cv-lab3/1.0 (educational)"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        images[name] = prepare_hr(Image.open(path))
    return images


def _half_masks(size):
    left = np.zeros((size, size), dtype=bool)
    right = np.zeros((size, size), dtype=bool)
    left[:, : size // 2] = True
    right[:, size // 2 :] = True
    return left, right


def _sheet(bicubic, mask, outputs, hr):
    panels = [
        ("bicubic", bicubic),
        ("маска", _mask_image(mask, bicubic.size)),
        (MASK_TITLE["zeros"], outputs["zeros"]),
        (MASK_TITLE["combined"], outputs["combined"]),
        (MASK_TITLE["ones"], outputs["ones"]),
        ("эталон", hr),
    ]
    tiles = [_labeled(image, title) for title, image in panels]
    width = sum(tile.width for tile in tiles)
    height = tiles[0].height
    sheet = Image.new("RGB", (width, height), "white")
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    return sheet


def _mask_image(mask, size):
    array = (mask * 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L").resize(size, Image.Resampling.NEAREST).convert("RGB")
    return image


def _labeled(image, title):
    bar = 36
    canvas = Image.new("RGB", (image.width, image.height + bar), "white")
    canvas.paste(image, (0, bar))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 20)
            break
    draw.text((8, 6), title, fill="black", font=font)
    return canvas


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows):
    buckets = {}
    for row in rows:
        key = (row["degradation"], row["mask"])
        buckets.setdefault(key, []).append(row)
    summary = []
    for deg_name in DEG_ORDER:
        for mask_name in MASK_ORDER:
            group = buckets[(deg_name, mask_name)]
            summary.append(
                {
                    "degradation": deg_name,
                    "mask": mask_name,
                    "psnr": _mean(group, "psnr"),
                    "lpips": _mean(group, "lpips"),
                    "clipiqa": _mean(group, "clipiqa"),
                    "psnr_left": _mean(group, "psnr_left"),
                    "psnr_right": _mean(group, "psnr_right"),
                }
            )
    return summary


def _mean(group, key):
    return float(np.mean([row[key] for row in group]))


def _chart(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = (("psnr", "PSNR, дБ", False), ("lpips", "LPIPS", True), ("clipiqa", "CLIP-IQA", False))
    colors = {"zeros": "#8d8d8d", "combined": "#e09f3e", "ones": "#2a9d8f"}
    x = np.arange(len(DEG_ORDER))
    width = 0.24
    for axis, (key, title, lower_better) in zip(axes, metrics):
        for index, mask_name in enumerate(MASK_ORDER):
            values = [
                next(row[key] for row in summary if row["degradation"] == deg and row["mask"] == mask_name)
                for deg in DEG_ORDER
            ]
            offset = (index - 1) * width
            axis.bar(x + offset, values, width, label=MASK_TITLE[mask_name], color=colors[mask_name])
        axis.set_xticks(x, [DEG_TITLE[name] for name in DEG_ORDER])
        axis.set_title(title + (" (меньше — лучше)" if lower_better else " (больше — лучше)"))
        axis.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
