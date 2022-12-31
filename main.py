from common import utils
from detection.detector import FaceDetector
from preprocessing.preprocessor import Preprocessor
from recognition.fisher import FisherRecognizer

IMAGE_DIR = "data/ziyad"
IMAGE_NAME = "image.png"
IMAGE_PATH = f"{IMAGE_DIR}/{IMAGE_NAME}"


if __name__ == "__main__":
    image = utils.read_as_float(IMAGE_PATH)
    scales = [0.25, 0.30, 0.35]

    detector = FaceDetector()
    detector.load()
    bb = detector.detect(image, scales)
    faces = (image[x1:x2, y1:y2] for x1, y1, x2, y2, *_ in bb)

    preprocessor = Preprocessor()
    faces = preprocessor.preprocess(faces)

    recognizer = FisherRecognizer()
    recognizer.load()
    predictions = recognizer.predict(faces)

    print(predictions)
