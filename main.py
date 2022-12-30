import numpy as np
from skimage import io, transform, util

from detection.detector import FaceDetector
from helpers import prepare_data
from recognition.constants import TESTING_PATH, TRAINING_PATH
from recognition.fisher import FisherRecognizer

IMAGE_DIR = "data/ziyad"
IMAGE_NAME = "image.png"
IMAGE_PATH = f"{IMAGE_DIR}/{IMAGE_NAME}"


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


if __name__ == "__main__":
    scales = [0.25, 0.30, 0.35, 0.40, 0.45]
    size = 39
    cutoff = 5

    training_faces, training_labels = prepare_data(
        TRAINING_PATH, scales, size=size, start=None, limit=cutoff
    )
    extra_faces, extra_labels = prepare_data(
        TRAINING_PATH, scales, size=size, start=cutoff, limit=None
    )
    testing_faces, testing_labels = prepare_data(
        TESTING_PATH, scales, size=size, start=None, limit=None
    )

    recognizer = FisherRecognizer()

    recognizer.fit(training_faces, training_labels)

    results = recognizer.predict(testing_faces)

    print(testing_labels)
    print(results)

    score = recognizer.score(testing_faces, testing_labels)
    print(score)

    results = recognizer.predict(extra_faces)

    print(extra_labels)
    print(results)

    score = recognizer.score(extra_faces, extra_labels)
    print(score)

    score = recognizer.score(
        np.concatenate((training_faces, extra_faces)),
        np.concatenate((training_labels, extra_labels)),
    )
    print(score)
