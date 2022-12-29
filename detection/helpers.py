import os

import numpy as np
from skimage import io, util
from sklearn import svm

from .hog import hog


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


def directory_hogs(directory: str) -> list[np.ndarray]:
    return [hog(read_as_float(os.path.join(directory, file))) for file in os.listdir(directory)]


def predict_with_score(clf: svm.SVC, vec: np.ndarray) -> tuple[int, float]:
    score = clf.decision_function([vec])[0]
    idx = int(score > 0)

    return clf.classes_[idx], score
