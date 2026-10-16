"""Метрики и полоски сравнения для уже посчитанных кадров SUPIR.

Считает их окружение baseline: там стоят PSNR, LPIPS и CLIP-IQA.
Кадры baseline не перезаписываются.
"""

from pathlib import Path

import torch
from PIL import Image

from degrade import HR_SIZE, make_masks
from experiment import (
    DEG_ORDER,
    MASK_ORDER,
    _chart,
    _half_masks,
    _sheet,
    _summarize,
    _write_csv,
)
from metrics_eval import load_metrics, region_psnr, score_pair


ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "results_supir"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = load_metrics(device)
    rows = []
    grids = source / "grids"
    grids.mkdir(parents=True, exist_ok=True)

    for name in ("cat", "bridge", "mountain"):
        hr = Image.open(source / "hr" / f"{name}.png").convert("RGB")
        for deg_name in DEG_ORDER:
            lq = Image.open(source / "lq" / f"{name}_{deg_name}.png").convert("RGB")
            bicubic = lq.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
            masks = make_masks(lq.height, lq.width)
            outputs = {}
            for mask_name in MASK_ORDER:
                prediction = Image.open(source / "outputs" / f"{name}_{deg_name}_{mask_name}.png").convert("RGB")
                outputs[mask_name] = prediction
                scores = score_pair(metrics, prediction, hr, device)
                left, right = _half_masks(HR_SIZE)
                scores["psnr_left"] = region_psnr(prediction, hr, left)
                scores["psnr_right"] = region_psnr(prediction, hr, right)
                rows.append({"image": name, "degradation": deg_name, "mask": mask_name, **scores})
                print(
                    f"{name} / {deg_name} / {mask_name}  "
                    f"PSNR {scores['psnr']:.2f}  LPIPS {scores['lpips']:.3f}  CLIP-IQA {scores['clipiqa']:.3f}",
                    flush=True,
                )
            base = score_pair(metrics, bicubic, hr, device)
            left, right = _half_masks(HR_SIZE)
            base["psnr_left"] = region_psnr(bicubic, hr, left)
            base["psnr_right"] = region_psnr(bicubic, hr, right)
            rows.append({"image": name, "degradation": deg_name, "mask": "bicubic", **base})
            _sheet(bicubic, masks["combined"], outputs, hr).save(grids / f"{name}_{deg_name}.jpg", quality=90)

    _write_csv(source / "metrics.csv", rows)
    summary = _summarize(rows)
    _write_csv(source / "summary.csv", summary)
    _chart(summary, source / "chart.png")
    print(f"Готово: {source}")


if __name__ == "__main__":
    main()
