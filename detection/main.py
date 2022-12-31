import pickle
from time import perf_counter_ns

import numpy as np
from skimage import color, io

from common import utils

from .detector import FaceDetector

IMAGE_DIR = "data/ziyad"
IMAGE_NAME = "image.png"


def detect_helper(detector: FaceDetector, path: str, scales: list[float]):
    img: np.ndarray = utils.read_as_float(path)

    start = perf_counter_ns()
    faces = detector.detect(img, scales)
    end = perf_counter_ns()
    print(f"Time: {(end - start) / 1e9} seconds")

    with open(f"{IMAGE_DIR}/faces.pkl", "wb") as fd:
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

    io.imsave(f"{IMAGE_DIR}/detected.jpg", img_original)

    return faces


if __name__ == "__main__":
    detector: FaceDetector = FaceDetector()
    detector.load()

    path: str = f"{IMAGE_DIR}/{IMAGE_NAME}"
    scales: list[float] = [0.25]
    # a smaller scale gets bigger faces

    detect_helper(detector, path, scales)
