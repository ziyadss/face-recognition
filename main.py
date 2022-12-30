import pickle

import numpy as np
from skimage import io
from cv2 import face

from detection.detector import BoxType

IMAGE_DIR = "data/ziyad"
IMAGE_NAME = "image.png"
IMAGE_PATH = f"{IMAGE_DIR}/{IMAGE_NAME}"

if __name__ == "__main__":
    img: np.ndarray = io.imread(IMAGE_PATH)
    with open(f"{IMAGE_DIR}/faces.pkl", "rb") as fd:
        faces: list[BoxType] = pickle.load(fd)

    cropped_faces = [img[x1:x2, y1:y2] for x1, y1, x2, y2, *_ in faces]

    recognizer = face.FisherFaceRecognizer_create()
