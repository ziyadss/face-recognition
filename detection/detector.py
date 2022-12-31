import multiprocessing
import pickle
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
from skimage import transform
from sklearn import metrics, svm

from .constants import *
from .hog import hog


class BoxType(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    scale: float


class FaceDetector:
    def __init__(self):
        self.clf: svm.SVC = svm.SVC()

    def dump(self, path: str = CLASSIFIER_PATH):
        with open(path, "wb") as fd:
            pickle.dump(self.clf, fd)

    def load(self, path: str = CLASSIFIER_PATH):
        with open(path, "rb") as fd:
            self.clf: svm.SVC = pickle.load(fd)

    def train(self, faces: Iterable[np.ndarray], non_faces: Iterable[np.ndarray]):
        faces_hog = [hog(face) for face in faces]
        non_faces_hog = [hog(non_face) for non_face in non_faces]

        train_data = faces_hog + non_faces_hog
        train_labels = [FACE] * len(faces_hog) + [NON_FACE] * len(non_faces_hog)

        self.clf.fit(train_data, train_labels)

    def _predict(self, vec):
        scores = self.clf.decision_function(vec)
        idx = (scores > BINARY_THRESHOLD).astype(int)
        return self.clf.classes_[idx], scores

    def test(self, faces: Iterable[np.ndarray], non_faces: Iterable[np.ndarray]):
        faces_hog = [hog(face) for face in faces]
        non_faces_hog = [hog(non_face) for non_face in non_faces]

        test_data = faces_hog + non_faces_hog
        test_labels = [FACE] * len(faces_hog) + [NON_FACE] * len(non_faces_hog)

        return self.clf.score(test_data, test_labels)

    @staticmethod
    def _iou(a: BoxType, b: BoxType) -> float:
        x11, y11, x12, y12, *_ = a
        x21, y21, x22, y22, *_ = b

        x_overlap: int = max(0, min(x12, x22) - max(x11, x21))
        y_overlap: int = max(0, min(y12, y22) - max(y11, y21))
        intersection: int = x_overlap * y_overlap

        area1: int = (x12 - x11) * (y12 - y11)
        area2: int = (x22 - x21) * (y22 - y21)
        union: int = area1 + area2 - intersection

        return intersection / union

    @staticmethod
    def _nms(B: list[BoxType]) -> list[BoxType]:
        res: list[BoxType] = []
        for bi in B:
            for bj in B:
                if __class__._iou(bi, bj) > NMS_THRESHOLD and bj[4] > bi[4]:
                    break
            else:
                res.append(bi)
        return res

    def _detect(self, img: np.ndarray, scale: float) -> list[BoxType]:
        scaled_img = transform.rescale(img, scale)

        h, w, *_ = scaled_img.shape
        HW, WW = WINDOW_SHAPE
        HS, WS = WINDOW_SHIFT

        boxes: list[BoxType] = []
        for startX in range(0, h - HW, HS):
            endX: int = startX + HW

            for startY in range(0, w - WW, WS):
                endY: int = startY + WW

                window: np.ndarray = scaled_img[startX:endX, startY:endY]
                hog_img: np.ndarray = hog(window)

                predictions, scores = self._predict([hog_img])
                prediction, score = predictions[0], scores[0]
                if prediction == FACE:
                    # boxes.append(BoxType(startX, startY, endX, endY, score, scale))
                    boxes.append(
                        BoxType(
                            int(np.round(startX / scale)),
                            int(np.round(startY / scale)),
                            int(np.round(endX / scale)),
                            int(np.round(endY / scale)),
                            score,
                            scale,
                        )
                    )

        # boxes = nms(boxes)
        # is this needed? only if nms is expensive, divide-and-conquer thing.
        # nms is O(n^2) worst case - n^2/1000 ms to be exact, decays to O(n) - n/1000 ms to be exact if full of discards.
        # (exact timings on my machine - Ziyad's Lenovo Legion)
        # so if expensive, it wont be discarding anyways, so no need to do it twice?
        # if not needed, remove loop below, have de-scaling done in previous loop. - doing this!

        # for i in range(len(boxes)):
        #     x1, y1, x2, y2, score, scale = boxes[i]
        #     boxes[i] = BoxType(
        #         int(np.round(x1 / scale)),
        #         int(np.round(y1 / scale)),
        #         int(np.round(x2 / scale)),
        #         int(np.round(y2 / scale)),
        #         score,
        #         scale,
        #     )

        return boxes

    def _detect_with_scales(
        self, img: np.ndarray, scales: list[float]
    ) -> list[BoxType]:
        with multiprocessing.Pool(PROCESSES_COUNT) as pool:
            params = ((img, scale) for scale in scales)
            nested_boxes = pool.starmap(self._detect, params)
        boxes = [box for boxes in nested_boxes for box in boxes]
        return self._nms(boxes)

    def detect(self, img: np.ndarray, scales: list[float] | float) -> list[BoxType]:
        if isinstance(scales, float):
            return __class__._nms(self._detect(img, scales))
        elif len(scales) == 1:
            return __class__._nms(self._detect(img, scales[0]))
        else:
            return self._detect_with_scales(img, scales)
