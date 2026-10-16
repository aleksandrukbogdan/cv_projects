"""PSNR, LPIPS и CLIP-IQA. Первые две сравнивают с эталоном, CLIP-IQA — без эталона."""

import numpy as np
import pyiqa
import torch
from PIL import Image


def load_metrics(device):
    return {
        "psnr": pyiqa.create_metric("psnr", device=device),
        "lpips": pyiqa.create_metric("lpips", device=device),
        "clipiqa": pyiqa.create_metric("clipiqa", device=device),
    }


def score_pair(metrics, prediction, reference, device):
    pred = _tensor(prediction, device)
    ref = _tensor(reference, device)
    if pred.shape != ref.shape:
        pred = torch.nn.functional.interpolate(pred, size=ref.shape[-2:], mode="bicubic", align_corners=False)
        pred = pred.clamp(0, 1)
    return {
        "psnr": _scalar(metrics["psnr"](pred, ref)),
        "lpips": _scalar(metrics["lpips"](pred, ref)),
        "clipiqa": _scalar(metrics["clipiqa"](pred)),
    }


def _scalar(value):
    if torch.is_tensor(value):
        value = value.detach().float().reshape(-1)[0].item()
    return float(value)


def region_psnr(prediction, reference, region):
    """region — булева маска HxW в разрешении эталона. Считается по пикселям области."""
    pred = np.asarray(prediction).astype(np.float32) / 255.0
    ref = np.asarray(reference).astype(np.float32) / 255.0
    if pred.shape[:2] != region.shape:
        image = Image.fromarray((pred * 255).astype(np.uint8))
        image = image.resize((region.shape[1], region.shape[0]), Image.Resampling.BICUBIC)
        pred = np.asarray(image).astype(np.float32) / 255.0
    diff = pred[region] - ref[region]
    mse = float(np.mean(diff ** 2))
    if mse < 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _tensor(image, device):
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)[None]
    return tensor.to(device=device)
