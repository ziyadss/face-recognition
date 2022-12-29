import pickle
from time import perf_counter_ns

import numpy as np
from skimage import io, color
from sklearn import svm

from constants import CLASSIFIER_PATH
from detection import BoxType, detect_with_scales
from helpers import read_as_float


def detect_helper(path: str, scales: list[float]):
    start = perf_counter_ns()

    img: np.ndarray = read_as_float(path)
    faces: list[BoxType] = detect_with_scales(clf, img, scales)

    end = perf_counter_ns()
    print(f"Time: {(end - start) / 1e9} seconds")

    with open("faces.pkl", "wb") as fd:
        pickle.dump(faces, fd)

    img_original: np.ndarray = io.imread(path)

    if img_original.ndim == 2:
        img_original = color.gray2rgb(img_original)
        img_original = np.stack([img_original] * 3, axis=2)

    for x1, y1, x2, y2, *_ in faces:
        img_original[x1, y1:y2, 0:3] = [255, 0, 0]
        img_original[x1:x2, y1, 0:3] = [255, 0, 0]
        img_original[x2, y1:y2, 0:3] = [255, 0, 0]
        img_original[x1:x2, y2, 0:3] = [255, 0, 0]

    io.imshow(img_original)
    io.show()

    io.imsave("detected.jpg", img_original)


if __name__ == "__main__":
    with open(CLASSIFIER_PATH, "rb") as fd:
        clf: svm.SVC = pickle.load(fd)

    path: str = "images/pic.jpg"
    scales: list[float] = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    # a smaller scale gets bigger faces

    detect_helper(path, scales)
