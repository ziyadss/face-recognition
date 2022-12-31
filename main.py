from skimage import io

from common import utils
from detection.detector import FaceDetector
from preprocessing.preprocessor import Preprocessor
from recognition.fisher import FisherRecognizer

IMAGE_DIR = "data"
IMAGE_NAME = "148056.jpg"
IMAGE_PATH = f"{IMAGE_DIR}/{IMAGE_NAME}"


if __name__ == "__main__":
    colored = io.imread(IMAGE_PATH)
    image = utils.read_as_float(IMAGE_PATH)
    scales = [0.15, 0.20, 0.25]

    detector = FaceDetector()
    detector.load()
    bb = detector.detect(image, scales)
    # bb = [bb[0]]
    print(bb)
    faces = [colored[x1:x2, y1:y2] for x1, y1, x2, y2, *_ in bb]
    for i, face in enumerate(faces):
        io.imsave(f"face_{i}.jpg", face)

    preprocessor = Preprocessor()
    faces = preprocessor.preprocess(faces)

    recognizer = FisherRecognizer()
    recognizer.load()
    predictions = recognizer.predict(faces)

    print(predictions)
