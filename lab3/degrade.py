"""Три деградации кадра 512×512 → вход модели 128×128.

bicubic — честный bicubic ×4, как дорожка DIV2K bicubic.
unknown — мягкая слепая деградация: размытие, уменьшение, шум, JPEG.
realesrgan — два прохода в духе Real-ESRGAN: размытие, ресайз, шум, JPEG.
"""

import cv2
import numpy as np
from PIL import Image


HR_SIZE = 512
SCALE = 4
LQ_SIZE = HR_SIZE // SCALE


def prepare_hr(image, size=HR_SIZE):
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def degrade_bicubic(hr):
    return hr.resize((LQ_SIZE, LQ_SIZE), Image.Resampling.BICUBIC)


def degrade_unknown(hr):
    array = np.array(hr)
    blurred = cv2.GaussianBlur(array, (0, 0), sigmaX=2.0)
    small = cv2.resize(blurred, (LQ_SIZE, LQ_SIZE), interpolation=cv2.INTER_CUBIC)
    noise = np.random.default_rng(0).normal(0, 10, small.shape)
    noisy = np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return _jpeg(noisy, quality=45)


def degrade_realesrgan(hr, seed=0):
    """Упрощённый двухкаскадный пайплайн Real-ESRGAN с фиксированным seed."""
    rng = np.random.default_rng(seed)
    image = np.array(hr).astype(np.float32) / 255.0
    target = LQ_SIZE

    for order in (0, 1):
        sigma = float(rng.uniform(0.2, 2.2 if order == 0 else 1.2))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        if order == 0:
            scale = float(rng.uniform(0.35, 0.7))
            new_w = max(target, int(image.shape[1] * scale))
            new_h = max(target, int(image.shape[0] * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)
        sigma_noise = float(rng.uniform(0, 18 if order == 0 else 8)) / 255.0
        image = image + rng.normal(0, sigma_noise, image.shape).astype(np.float32)
        image = np.clip(image, 0, 1)
        quality = int(rng.uniform(35, 75) if order == 0 else rng.uniform(45, 90))
        image = _jpeg_float(image, quality)

    if image.shape[0] != target or image.shape[1] != target:
        image = cv2.resize(image, (target, target), interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def make_degradations(hr, seed=0):
    return {
        "bicubic": degrade_bicubic(hr),
        "unknown": degrade_unknown(hr),
        "realesrgan": degrade_realesrgan(hr, seed=seed),
    }


def make_masks(height, width):
    """Маска в разрешении входа. 1 — восстанавливать диффузией, 0 — оставить bicubic."""
    ones = np.ones((height, width), dtype=np.float32)
    zeros = np.zeros((height, width), dtype=np.float32)
    combined = np.zeros((height, width), dtype=np.float32)
    combined[:, : width // 2] = 1.0
    return {"ones": ones, "zeros": zeros, "combined": combined}


def _jpeg(array_rgb, quality):
    bgr = cv2.cvtColor(array_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Не удалось закодировать JPEG")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def _jpeg_float(image, quality):
    array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.array(_jpeg(array, quality)).astype(np.float32) / 255.0
