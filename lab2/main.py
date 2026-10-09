import argparse
import sys
from pathlib import Path

import cv2

from tracker import ObjectTracker, draw_box


# Кадры, которые кладём в отчёт.
SHOT_FRAMES = [0, 18, 80, 200, 400, 520]


def default_videos(base_dir):
    data_dir = base_dir.parent / "data"
    names = [
        "mona-lisa.avi",
        "mona-lisa-blur.avi",
        "mona-lisa-blur-extra-credit.avi",
    ]
    return [data_dir / name for name in names]


def open_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if writer.isOpened():
        return writer
    # Если mp4 на этой сборке OpenCV не пишется, сохраняем avi.
    avi_path = path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(avi_path), fourcc, fps, size)
    return writer


def process_video(video_path, results_dir, label):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return None

    ok, first = capture.read()
    if not ok:
        print(f"В видео нет кадров: {video_path}")
        capture.release()
        return None

    tracker = ObjectTracker(first)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    shots_dir = results_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    # Не писать результат в тот же файл, что и исходное видео.
    out_path = results_dir / f"{video_path.stem}_tracked.mp4"
    if out_path.resolve() == video_path.resolve():
        out_path = results_dir / f"{video_path.stem}_out.mp4"
    writer = open_writer(out_path, fps, (width, height))

    found = 0
    index = 0
    frame = first
    print(f"{video_path.name}: кадров {frame_count}, размер {width}x{height}")

    while True:
        view = frame.copy()
        corners = tracker.locate(frame)
        if corners is not None:
            found += 1
            draw_box(view, corners, label)

        if index in SHOT_FRAMES:
            shot_path = shots_dir / f"{video_path.stem}_{index:04d}.jpg"
            cv2.imwrite(str(shot_path), view)

        writer.write(view)
        index += 1
        if index % 100 == 0:
            print(f"  кадр {index}")

        ok, frame = capture.read()
        if not ok:
            break

    capture.release()
    writer.release()

    missed = index - found
    print(f"  рамка есть на {found} кадрах из {index}, пропущено {missed}")
    print(f"  видео: {out_path}")
    return {
        "name": video_path.name,
        "frames": index,
        "found": found,
        "missed": missed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Трекинг плоского объекта по ключевым точкам"
    )
    parser.add_argument(
        "videos",
        nargs="*",
        help="Видео. Если не указаны, берутся три тестовых ролика из data/",
    )
    parser.add_argument(
        "--label",
        default="Mona Lisa",
        help="Подпись на рамке. Шрифт OpenCV рисует латиницу",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if args.videos:
        videos = [Path(path) for path in args.videos]
    else:
        videos = default_videos(base_dir)

    rows = []
    for video_path in videos:
        if not video_path.exists():
            print(f"Файл не найден: {video_path}")
            return 1
        row = process_video(video_path, results_dir, args.label)
        if row is None:
            return 1
        rows.append(row)

    print()
    print("name\tframes\tfound\tmissed")
    lines = ["name\tframes\tfound\tmissed"]
    for row in rows:
        line = f"{row['name']}\t{row['frames']}\t{row['found']}\t{row['missed']}"
        print(line)
        lines.append(line)
    (results_dir / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
