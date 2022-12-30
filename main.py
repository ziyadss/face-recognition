import numpy as np
from skimage import io, util

from detection.detector import FaceDetector
from preprocessing.preprocessor import Preprocessor, PreprocessorConfiguration
from recognition.fisher import FisherRecognizer

IMAGE_DIR = "data/ziyad"
IMAGE_NAME = "image.png"
IMAGE_PATH = f"{IMAGE_DIR}/{IMAGE_NAME}"


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


if __name__ == "__main__":
    image = read_as_float(IMAGE_PATH)
    scales = [0.25, 0.30, 0.35]

    detector = FaceDetector()
    bb = detector.detect(image, scales)
    faces = detector.extract(image, bb)

    size = (39, 39)
    preprocessor_config = PreprocessorConfiguration(output_shape=size)
    preprocessor = Preprocessor(preprocessor_config)
    faces = preprocessor.preprocess(faces)

    recognizer = FisherRecognizer(load=True)
    predictions = recognizer.predict(faces)

    print(predictions)
