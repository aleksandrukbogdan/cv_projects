import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sobel import (
    normalize_magnitude,
    normalize_signed,
    sobel_opencv,
    sobel_python,
    to_grayscale,
)


# Своя реализация на чистых циклах медленная.
# Если сторона больше этого значения, картинку уменьшаем, иначе ждать придётся минуты.
MAX_SIDE = 500

# Один прогон скачет из-за загрузки процессора, поэтому берём среднее.
# OpenCV считается миллисекунды, ему нужно больше повторов, чем циклам на Python.
PYTHON_REPEATS = 3
OPENCV_REPEATS = 10


def create_test_image(path):
    """Простая картинка с фигурами: на ней хорошо видно, где Собель находит границы."""
    image = np.full((300, 400, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (40, 50), (170, 190), (40, 40, 180), thickness=-1)
    cv2.circle(image, (290, 130), 70, (40, 140, 50), thickness=-1)
    cv2.line(image, (30, 240), (370, 240), (30, 30, 30), thickness=4)
    cv2.putText(
        image,
        "Sobel",
        (70, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (70, 70, 70),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), image)


def limit_size(image):
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / longest
    new_size = (int(width * scale), int(height * scale))
    print(
        f"Картинка {width}x{height} уменьшена до {new_size[0]}x{new_size[1]}: "
        "своя реализация на циклах иначе считается очень долго."
    )
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def measure(func, gray, repeats):
    """Возвращает результат и среднее время одного вызова."""
    result = None
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(gray)
        total += time.perf_counter() - start
    return result, total / repeats


def save_timing_chart(rows, path):
    labels = [row["label"] for row in rows]
    python_times = [row["python"] for row in rows]
    opencv_times = [row["opencv"] for row in rows]

    x = np.arange(len(labels))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - bar_width / 2, python_times, bar_width, label="Своя реализация")
    ax.bar(x + bar_width / 2, opencv_times, bar_width, label="OpenCV")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Время, секунды (логарифмическая шкала)")
    ax.set_xlabel("Размер изображения")
    ax.set_title("Сравнение времени работы фильтра Собеля")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def gray_to_bgr(gray_u8):
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def with_caption(image_bgr, text):
    """Подпись сверху. Шрифт OpenCV не умеет кириллицу, поэтому подписи на английском."""
    header = np.full((32, image_bgr.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, image_bgr])


def print_timing(label, python_time, opencv_time):
    speedup = python_time / opencv_time if opencv_time > 0 else 0
    print(
        f"{label}: своя реализация {python_time:.4f} с, "
        f"OpenCV {opencv_time:.6f} с, "
        f"OpenCV быстрее в {speedup:.0f} раз"
    )


def run_benchmark(gray):
    """Считает фильтр на трёх размерах, от меньшего к большему.

    На полном размере заодно запоминаем сами картинки градиента,
    чтобы не запускать фильтр ещё раз отдельно от замера.
    """
    height, width = gray.shape
    # Последний элемент — исходный размер, его результат пойдёт в картинки.
    samples = []
    for scale in (0.35, 0.6, 1.0):
        if scale == 1.0:
            samples.append((gray, True))
        else:
            new_width = max(3, int(width * scale))
            new_height = max(3, int(height * scale))
            small = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            samples.append((small, False))

    rows = []
    full_python = None
    full_opencv = None

    for sample, is_full_size in samples:
        sample_height, sample_width = sample.shape
        label = f"{sample_width}x{sample_height}"
        print(f"Замер {label}...")
        python_result, python_time = measure(sobel_python, sample, repeats=PYTHON_REPEATS)
        opencv_result, opencv_time = measure(sobel_opencv, sample, repeats=OPENCV_REPEATS)
        rows.append(
            {
                "label": label,
                "python": python_time,
                "opencv": opencv_time,
            }
        )
        print_timing(label, python_time, opencv_time)
        if is_full_size:
            full_python = python_result
            full_opencv = opencv_result

    return rows, full_python, full_opencv


def main():
    parser = argparse.ArgumentParser(description="Фильтр Собеля: своя реализация и OpenCV")
    parser.add_argument(
        "--image",
        default=None,
        help="Путь к изображению. Если не указан, берётся images/test.png",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    images_dir = base_dir / "images"
    results_dir = base_dir / "results"
    images_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = images_dir / "test.png"
        if not image_path.exists():
            create_test_image(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Не удалось открыть изображение: {image_path}")
        return 1

    image = limit_size(image)
    gray = to_grayscale(image)
    height, width = gray.shape
    print(f"Размер для расчёта: {width}x{height}")
    print(
        f"Своя реализация усредняется по {PYTHON_REPEATS} запускам, "
        f"OpenCV — по {OPENCV_REPEATS}."
    )
    print()

    rows, (gx_py, gy_py, mag_py), (_, _, mag_cv) = run_benchmark(gray)
    python_time = rows[-1]["python"]
    opencv_time = rows[-1]["opencv"]

    # Своя версия не трогает рамку в 1 пиксель. OpenCV на краю достраивает пиксели
    # за границей, поэтому там результат может отличаться. Сравниваем внутреннюю часть.
    inner_py = mag_py[1 : height - 1, 1 : width - 1]
    inner_cv = mag_cv[1 : height - 1, 1 : width - 1]
    abs_diff = np.abs(inner_py - inner_cv)
    mean_diff = float(abs_diff.mean())
    max_diff = float(abs_diff.max())

    mag_py_u8 = normalize_magnitude(mag_py)
    mag_cv_u8 = normalize_magnitude(mag_cv)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    cv2.imwrite(str(results_dir / "gray.png"), gray_u8)
    cv2.imwrite(str(results_dir / "gx_python.png"), normalize_signed(gx_py))
    cv2.imwrite(str(results_dir / "gy_python.png"), normalize_signed(gy_py))
    cv2.imwrite(str(results_dir / "sobel_python.png"), mag_py_u8)
    cv2.imwrite(str(results_dir / "sobel_opencv.png"), mag_cv_u8)

    diff_u8 = np.clip(abs_diff, 0, 255).astype(np.uint8)
    diff_full = np.zeros((height, width), dtype=np.uint8)
    diff_full[1 : height - 1, 1 : width - 1] = diff_u8
    cv2.imwrite(str(results_dir / "difference.png"), diff_full)

    comparison = np.hstack(
        [
            with_caption(image, "original"),
            with_caption(gray_to_bgr(gray_u8), "gray"),
            with_caption(gray_to_bgr(mag_py_u8), "python"),
            with_caption(gray_to_bgr(mag_cv_u8), "opencv"),
        ]
    )
    cv2.imwrite(str(results_dir / "comparison.png"), comparison)

    print()
    print(f"Средняя |разница| внутри изображения: {mean_diff:.3e}")
    print(f"Максимальная |разница| внутри изображения: {max_diff:.3e}")

    save_timing_chart(rows, results_dir / "timing.png")

    lines = [
        f"image: {image_path}",
        f"size: {width}x{height}",
        f"python_seconds: {python_time:.6f}",
        f"opencv_seconds: {opencv_time:.6f}",
        f"python_repeats: {PYTHON_REPEATS}",
        f"opencv_repeats: {OPENCV_REPEATS}",
        f"mean_abs_diff_inner: {mean_diff:.3e}",
        f"max_abs_diff_inner: {max_diff:.3e}",
        "",
        "size\tpython_s\topencv_s",
    ]
    for row in rows:
        lines.append(f"{row['label']}\t{row['python']:.6f}\t{row['opencv']:.6f}")
    (results_dir / "timings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Результаты сохранены в {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
