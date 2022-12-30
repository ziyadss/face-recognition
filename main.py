import numpy as np
from skimage import io, util

from detection.detector import FaceDetector
from helpers import prepare_data
from preprocessing.preprocessor import Preprocessor, PreprocessorConfiguration
from recognition.constants import TESTING_PATH, TRAINING_PATH
from recognition.fisher import FisherRecognizer

IMAGE_DIR = "data/ziyad"
IMAGE_NAME = "image.png"
IMAGE_PATH = f"{IMAGE_DIR}/{IMAGE_NAME}"


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


if __name__ == "__main__":
    cutoff = 5

    training_images, training_labels = prepare_data(TRAINING_PATH, take=cutoff)
    extra_images, extra_labels = prepare_data(TRAINING_PATH, skip=cutoff)
    testing_images, testing_labels = prepare_data(TESTING_PATH)

    detector = FaceDetector()
    scales = [0.25, 0.30, 0.35]
    training_faces_bb = [
        (image, detector.detect_one(image, scales)) for image in training_images
    ]
    extra_faces_bb = [
        (image, detector.detect_one(image, scales)) for image in extra_images
    ]
    testing_faces_bb = [
        (image, detector.detect_one(image, scales)) for image in testing_images
    ]

    training_faces = [
        image[x1:x2, y1:y2] for image, (x1, y1, x2, y2, *_) in training_faces_bb
    ]
    extra_faces = [
        image[x1:x2, y1:y2] for image, (x1, y1, x2, y2, *_) in extra_faces_bb
    ]
    testing_faces = [
        image[x1:x2, y1:y2] for image, (x1, y1, x2, y2, *_) in testing_faces_bb
    ]

    size = (39, 39)
    preprocessor_config = PreprocessorConfiguration(output_shape=size)
    preprocessor = Preprocessor(preprocessor_config)
    training_faces = preprocessor.preprocess(training_faces)
    extra_faces = preprocessor.preprocess(extra_faces)
    testing_faces = preprocessor.preprocess(testing_faces)

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
        training_faces + extra_faces, training_labels + extra_labels
    )
    print(score)
