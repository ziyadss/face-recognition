import numpy as np
from skimage import io, transform, util

from detection.detector import FaceDetector
from recognition.fisher import fisherfaces

IMAGE_DIR = "data/ziyad"
IMAGE_NAME = "image.png"
IMAGE_PATH = f"{IMAGE_DIR}/{IMAGE_NAME}"


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


if __name__ == "__main__":
    detector = FaceDetector()

    img: np.ndarray = read_as_float(IMAGE_PATH)
    scales = [0.25, 0.30, 0.35, 0.40, 0.45]

    faces = detector.detect(img, scales)

    cropped_faces = [img[x1:x2, y1:y2] for x1, y1, x2, y2, *_ in faces]

    RECOGNIZER_SIZE = (39, 39)
    cropped_faces = [transform.resize(face, RECOGNIZER_SIZE) for face in cropped_faces]

    A = "Alvaro Uribe"
    B = "George W Bush"
    C = "Amelia Vega"

    labels = [A, A, A, B, B, B, C, C, C]

    train_idx = [0, 1, 3, 4, 6, 7]
    test_idx = [2, 5, 8]

    train_faces = np.array([cropped_faces[i].flatten() for i in train_idx])
    train_labels = np.array([labels[i] for i in train_idx])

    test_faces = np.array([cropped_faces[i].flatten() for i in test_idx])
    test_labels = np.array([labels[i] for i in test_idx])

    recognizer = fisherfaces()

    recognizer.fit(train_faces, train_labels)

    results = recognizer.predict(test_faces)

    print(test_labels)
    print(results)

    score = recognizer.score(test_faces, test_labels)
    print(score)
