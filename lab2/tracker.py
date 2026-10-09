import cv2
import numpy as np


# Сколько совпадений с эталоном нужно, чтобы поверить новой рамке.
MIN_MATCH_INLIERS = 12
# Сколько точек оптического потока нужно, чтобы продолжить рамку со прошлого кадра.
MIN_FLOW_INLIERS = 8


class ObjectTracker:
    """Ищет плоский объект с первого кадра на остальных кадрах видео.

    Сначала сравниваем ключевые точки кадра с эталоном (первый кадр).
    Если объект далеко и точек мало, двигаем уже найденные точки
    оптическим потоком с предыдущего кадра.
    """

    def __init__(self, reference_bgr):
        self.reference = reference_bgr
        self.height, self.width = reference_bgr.shape[:2]
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.ref_keypoints, self.ref_descriptors = self.sift.detectAndCompute(reference_bgr, None)
        if self.ref_descriptors is None:
            raise RuntimeError("На первом кадре не нашлось ключевых точек")

        self.corners = np.float32([
            [0, 0],
            [self.width - 1, 0],
            [self.width - 1, self.height - 1],
            [0, self.height - 1],
        ]).reshape(-1, 1, 2)

        self.prev_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        # Одни и те же точки: координаты на эталоне и на прошлом кадре.
        self.ref_points = None
        self.cur_points = None

    def locate(self, frame_bgr):
        """Вернуть 4 угла объекта на кадре или None, если объект не виден."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        corners = self._match_reference(frame_bgr)
        if corners is None:
            corners = self._track_flow(gray)
        # Точки должны лежать на prev_gray. Если кадр потеряли,
        # старые координаты больше не совпадают со следующим кадром.
        if corners is None:
            self.ref_points = None
            self.cur_points = None

        self.prev_gray = gray
        return corners

    def _match_reference(self, frame_bgr):
        keypoints, descriptors = self.sift.detectAndCompute(frame_bgr, None)
        if descriptors is None or len(keypoints) < 2:
            return None

        pairs = self.matcher.knnMatch(self.ref_descriptors, descriptors, k=2)
        good = []
        for pair in pairs:
            # У слабой точки второй кандидат почти такой же близкий.
            # Такие совпадения выбрасываем (тест Лоу).
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < 0.75 * second.distance:
                good.append(best)

        if len(good) < MIN_MATCH_INLIERS:
            return None

        src = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if homography is None or mask is None:
            return None
        if int(mask.sum()) < MIN_MATCH_INLIERS:
            return None

        corners = cv2.perspectiveTransform(self.corners, homography)
        if not box_is_ok(corners, self.width, self.height):
            return None

        inliers = mask.reshape(-1).astype(bool)
        self.ref_points = src[inliers]
        self.cur_points = dst[inliers]
        return corners

    def _track_flow(self, gray):
        if self.cur_points is None or len(self.cur_points) < MIN_FLOW_INLIERS:
            return None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.cur_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        status = status.reshape(-1).astype(bool)
        if int(status.sum()) < MIN_FLOW_INLIERS:
            return None

        src = self.ref_points[status]
        dst = next_points[status]
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if homography is None or mask is None:
            return None
        if int(mask.sum()) < MIN_FLOW_INLIERS:
            return None

        corners = cv2.perspectiveTransform(self.corners, homography)
        if not box_is_ok(corners, self.width, self.height):
            return None

        inliers = mask.reshape(-1).astype(bool)
        self.ref_points = src[inliers]
        self.cur_points = dst[inliers]
        return corners


def box_is_ok(corners, width, height):
    """Отсечь четырёхугольник, который не может быть рамкой объекта."""
    quad = corners.reshape(4, 2).astype(np.float32)
    if not cv2.isContourConvex(quad):
        return False

    area = abs(cv2.contourArea(quad))
    frame_area = float(width * height)
    # Слишком мелкая рамка — шум. Слишком большая — гомография сломалась.
    if area < 1200 or area > frame_area * 1.05:
        return False

    if quad[:, 0].min() < -width * 0.25 or quad[:, 0].max() > width * 1.25:
        return False
    if quad[:, 1].min() < -height * 0.25 or quad[:, 1].max() > height * 1.25:
        return False
    return True


def draw_box(frame, corners, label):
    points = np.int32(corners).reshape(-1, 2)
    # Чёрная подложка, чтобы рамка была видна и на светлом столе, и на обложке.
    cv2.polylines(frame, [points], True, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.polylines(frame, [points], True, (40, 220, 40), 4, cv2.LINE_AA)

    top_index = int(np.argmin(points[:, 1]))
    x = int(points[top_index, 0])
    y = int(points[top_index, 1]) - 8
    y = max(22, min(y, frame.shape[0] - 8))
    x = max(0, min(x, frame.shape[1] - 140))
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 220, 40),
        2,
        cv2.LINE_AA,
    )
