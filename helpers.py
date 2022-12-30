import os
from typing import Optional

import cv2
import numpy as np
from skimage import io, util

from recognition.constants import CLASSIFIER_PATH


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


def detect_face_opencv(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(CLASSIFIER_PATH)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return image[y : y + w, x : x + h], faces[0]


def prepare_data(
    path: str,
    skip: Optional[int] = None,
    take: Optional[int] = None,
) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    labels: list[str] = []

    for dir in os.listdir(path):
        label = dir
        images_path = os.path.join(path, dir)
        images_names = os.listdir(images_path)

        for image_name in images_names[skip:take]:
            image_path = os.path.join(images_path, image_name)
            image = read_as_float(image_path)

            images.append(image)
            labels.append(label)

    return images, labels
