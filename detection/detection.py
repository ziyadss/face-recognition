import multiprocessing
from typing import NamedTuple

import numpy as np
from skimage import transform
from sklearn import svm

from constants import *
from helpers import predict_with_score
from hog import hog


class BoxType(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    scale: float


def iou(a: BoxType, b: BoxType) -> float:
    x11, y11, x12, y12, *_ = a
    x21, y21, x22, y22, *_ = b

    x_overlap: int = max(0, min(x12, x22) - max(x11, x21))
    y_overlap: int = max(0, min(y12, y22) - max(y11, y21))
    intersection: int = x_overlap * y_overlap

    area1: int = (x12 - x11) * (y12 - y11)
    area2: int = (x22 - x21) * (y22 - y21)
    union: int = area1 + area2 - intersection

    return intersection / union


def nms(B: list[BoxType]) -> list[BoxType]:
    res: list[BoxType] = []
    for bi in B:
        for bj in B:
            if iou(bi, bj) > NMS_THRESHOLD and bj[4] > bi[4]:
                break
        else:
            res.append(bi)
    return res


def detect(clf: svm.SVC, img: np.ndarray, scale: float) -> list[BoxType]:
    scaled_img = transform.rescale(img, scale)
    faces: list[BoxType] = []
    for startX in range(0, scaled_img.shape[0] - WINDOW_SHAPE[0], WINDOW_SHIFT[0]):
        endX: int = startX + WINDOW_SHAPE[0]

        for startY in range(0, scaled_img.shape[1] - WINDOW_SHAPE[1], WINDOW_SHIFT[1]):
            endY: int = startY + WINDOW_SHAPE[1]

            window: np.ndarray = scaled_img[startX:endX, startY:endY]
            hog_img: np.ndarray = hog(window)

            prediction, score = predict_with_score(clf, hog_img)
            if prediction == FACE and score > BINARY_THRESHOLD:
                faces.append(BoxType(startX, startY, endX, endY, score, scale))

    faces = nms(faces)
    for i in range(len(faces)):
        x1, y1, x2, y2, score, scale = faces[i]
        faces[i] = BoxType(
            int(np.round(x1 / scale)),
            int(np.round(y1 / scale)),
            int(np.round(x2 / scale)),
            int(np.round(y2 / scale)),
            score,
            scale,
        )

    return faces


def detect_with_scales(clf: svm.SVC, img: np.ndarray, scales: list[float]) -> list[BoxType]:
    with multiprocessing.Pool(PROCESSES_COUNT) as pool:
        params = ((clf, img, scale) for scale in scales)
        nested_boxes = pool.starmap(detect, params)
    boxes = [box for boxes in nested_boxes for box in boxes]

    boxes = nms(boxes)
    print(f"Found {len(boxes)} boxes")

    return boxes
