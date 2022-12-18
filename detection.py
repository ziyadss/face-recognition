import numpy as np
from skimage import transform
from sklearn import svm

from constants import *
from helpers import predict_with_score
from hog import hog

face_type = tuple[int, int, int, int, float, float]


def iou(a: face_type, b: face_type) -> float:
    x11, y11, x12, y12, *_ = a
    x21, y21, x22, y22, *_ = b

    x_overlap: int = max(0, min(x12, x22) - max(x11, x21))
    y_overlap: int = max(0, min(y12, y22) - max(y11, y21))
    intersection: int = x_overlap * y_overlap

    area1: int = (x12 - x11) * (y12 - y11)
    area2: int = (x22 - x21) * (y22 - y21)
    union: int = area1 + area2 - intersection

    return intersection / union


def nms(B: list[face_type], threshold: float) -> list[face_type]:
    res: list[face_type] = []
    for bi in B:
        discard: bool = False
        for bj in B:
            if iou(bi, bj) > threshold:
                if bj[4] > bi[4]:
                    discard = True
        if not discard:
            res.append(bi)
    return res


def detect(clf: svm.SVC, img: np.ndarray, scale: float = 1) -> list[face_type]:
    faces: list[face_type] = []
    for startX in range(0, img.shape[0] - WINDOW_SHAPE[0], WINDOW_SHIFT[0]):
        endX: int = startX + WINDOW_SHAPE[0]

        for startY in range(0, img.shape[1] - WINDOW_SHAPE[1], WINDOW_SHIFT[1]):
            endY: int = startY + WINDOW_SHAPE[1]

            window: np.ndarray = img[startX:endX, startY:endY]
            hog_img: np.ndarray = hog(window)

            prediction, score = predict_with_score(clf, hog_img)
            if prediction == FACE:
                faces.append((startX, startY, endX, endY, score, scale))
    return faces


def detect_with_scales(clf: svm.SVC, img: np.ndarray, scales: list[float]) -> list[face_type]:
    faces: list[face_type] = []
    for scale in scales:
        scaled_img = transform.rescale(img, scale)
        scale_faces = detect(clf, scaled_img, scale)
        faces.extend(scale_faces)

    print(f"Found: {len(faces)} faces")

    thresholded_faces: list = [face for face in faces if face[4] > BINARY_THRESHOLD]
    print(f"After binary thresholding: {len(thresholded_faces)} faces.")

    for i in range(len(thresholded_faces)):
        x1, y1, x2, y2, score, scale = thresholded_faces[i]
        thresholded_faces[i] = (
            int(np.round(x1 / scale)),
            int(np.round(y1 / scale)),
            int(np.round(x2 / scale)),
            int(np.round(y2 / scale)),
            score,
            scale,
        )

    faces: list[face_type] = nms(thresholded_faces, NMS_THRESHOLD)
    print(f"After NMS: {len(faces)} faces.")

    return faces
