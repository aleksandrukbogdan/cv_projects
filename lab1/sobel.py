import math

import cv2
import numpy as np


# Ядра Собеля 3x3.
# GX реагирует на вертикальные границы (изменение яркости по горизонтали),
# GY — на горизонтальные границы (изменение яркости по вертикали).
GX = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
]

GY = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
]


def to_grayscale(image_bgr):
    """Перевод BGR-картинки в яркость. Формула как в OpenCV."""
    blue = image_bgr[:, :, 0].astype(np.float64)
    green = image_bgr[:, :, 1].astype(np.float64)
    red = image_bgr[:, :, 2].astype(np.float64)
    return 0.114 * blue + 0.587 * green + 0.299 * red


def sobel_python(gray):
    """Фильтр Собеля своими циклами, без cv2.Sobel.

    Для каждого пикселя берём окно 3x3 и умножаем его на ядра GX и GY.
    Край в 1 пиксель не считаем: окно вылезло бы за границу картинки.
    """
    height, width = gray.shape
    gx_image = np.zeros((height, width), dtype=np.float64)
    gy_image = np.zeros((height, width), dtype=np.float64)
    magnitude = np.zeros((height, width), dtype=np.float64)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = 0.0
            gy = 0.0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    pixel = gray[y + ky, x + kx]
                    gx += pixel * GX[ky + 1][kx + 1]
                    gy += pixel * GY[ky + 1][kx + 1]
            gx_image[y, x] = gx
            gy_image[y, x] = gy
            magnitude[y, x] = math.sqrt(gx * gx + gy * gy)

    return gx_image, gy_image, magnitude


def sobel_opencv(gray):
    """Тот же фильтр через встроенный cv2.Sobel.

    Модуль градиента считаем сами, той же формулой, что и в sobel_python,
    чтобы сравнивать именно оператор Собеля, а не разные способы нормировки.
    """
    gx_image = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy_image = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx_image * gx_image + gy_image * gy_image)
    return gx_image, gy_image, magnitude


def normalize_magnitude(image):
    """Модуль градиента (числа от 0 и выше) в картинку 0..255."""
    max_value = image.max()
    if max_value == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = image / max_value * 255.0
    return scaled.astype(np.uint8)


def normalize_signed(image):
    """Производная со знаком: 0 — отрицательная, 128 — ноль, 255 — положительная."""
    max_abs = np.max(np.abs(image))
    if max_abs == 0:
        return np.full(image.shape, 128, dtype=np.uint8)
    scaled = image / max_abs * 127.0 + 128.0
    return np.clip(scaled, 0, 255).astype(np.uint8)
