import numpy as np
from skimage import transform
from sklearn import svm

from constants import *
from helpers import predict_with_score
from hog import hog

box_type = tuple[int, int, int, int, float, float]


def iou(a: box_type, b: box_type) -> float:
    x11, y11, x12, y12, *_ = a
    x21, y21, x22, y22, *_ = b

    x_overlap: int = max(0, min(x12, x22) - max(x11, x21))
    y_overlap: int = max(0, min(y12, y22) - max(y11, y21))
    intersection: int = x_overlap * y_overlap

    area1: int = (x12 - x11) * (y12 - y11)
    area2: int = (x22 - x21) * (y22 - y21)
    union: int = area1 + area2 - intersection

    return intersection / union


def nms(B: list[box_type]) -> list[box_type]:
    res: list[box_type] = []
    for bi in B:
        discard: bool = False
        for bj in B:
            if iou(bi, bj) > NMS_THRESHOLD:
                if bj[4] > bi[4]:
                    discard = True
        if not discard:
            res.append(bi)
    return res


def detect(clf: svm.SVC, img: np.ndarray, scale: float = 1) -> list[box_type]:
    faces: list[box_type] = []
    for startX in range(0, img.shape[0] - WINDOW_SHAPE[0], WINDOW_SHIFT[0]):
        endX: int = startX + WINDOW_SHAPE[0]

        for startY in range(0, img.shape[1] - WINDOW_SHAPE[1], WINDOW_SHIFT[1]):
            endY: int = startY + WINDOW_SHAPE[1]

            window: np.ndarray = img[startX:endX, startY:endY]
            hog_img: np.ndarray = hog(window)

            prediction, score = predict_with_score(clf, hog_img)
            if prediction == FACE and score > BINARY_THRESHOLD:
                faces.append((startX, startY, endX, endY, score, scale))

    return nms(faces)


def detect_with_scales(clf: svm.SVC, img: np.ndarray, scales: list[float]) -> list[box_type]:
    boxes: list[box_type] = []
    for scale in scales:
        scaled_img = transform.rescale(img, scale)
        scale_boxes = detect(clf, scaled_img, scale)
        boxes.extend(scale_boxes)

    for i in range(len(boxes)):
        x1, y1, x2, y2, score, scale = boxes[i]
        boxes[i] = (
            int(np.round(x1 / scale)),
            int(np.round(y1 / scale)),
            int(np.round(x2 / scale)),
            int(np.round(y2 / scale)),
            score,
            scale,
        )

    boxes = nms(boxes)
    print(f"Found {len(boxes)} boxes")

    return boxes
