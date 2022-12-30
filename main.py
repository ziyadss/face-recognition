import numpy as np
from skimage import io

from detection.detector import FaceDetector

IMAGE_DIR = "data/ziyad"
IMAGE_NAME = "image.png"
IMAGE_PATH = f"{IMAGE_DIR}/{IMAGE_NAME}"

if __name__ == "__main__":
    detector = FaceDetector()

    img: np.ndarray = io.imread(IMAGE_PATH)
    scales = [0.25, 0.30, 0.35, 0.40, 0.45]

    faces = detector.detect(img, scales)

    cropped_faces = [img[x1:x2, y1:y2] for x1, y1, x2, y2, *_ in faces]
