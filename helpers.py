import os

import numpy as np
from skimage import io, util
from sklearn import svm

from hog import hog


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


def directory_hogs(directory: str) -> list[np.ndarray]:
    return [
        hog(read_as_float(os.path.join(directory, file)))
        for file in os.listdir(directory)
    ]


def predict_with_scores(clf: svm.SVC, vecs) -> tuple[np.ndarray, np.ndarray]:
    scores = clf.decision_function(vecs)

    if len(scores.shape) == 1:
        indices = (scores > 0).astype(int)
    else:
        indices = scores.argmax(axis=1)

    return clf.classes_[indices], scores
