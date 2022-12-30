import os
from typing import Optional

import cv2
import numpy as np
from skimage import io, transform, util

from detection.detector import FaceDetector
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
    scales: list[float],
    size: tuple[int, int] = (21, 21),
    start: Optional[int] = None,
    limit: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    detector = FaceDetector()
    detected_faces: list[np.ndarray] = []
    face_labels: list[str] = []
    image_dirs = os.listdir(path)
    for dir_name in image_dirs:
        label = dir_name
        images_path = path + "/" + dir_name
        images_names = os.listdir(images_path)

        for image_name in images_names[start:limit]:
            image_path = images_path + "/" + image_name
            # image = cv2.imread(image_path)
            # face, _ = detect_face_opencv(image)
            image = read_as_float(image_path)
            detected = detector.detect(image, scales)
            faces = [image[x1:x2, y1:y2] for x1, y1, x2, y2, *_ in detected]
            # if face is not None:
            for face in faces:
                resized_face = transform.resize(face, size)
                detected_faces.append(resized_face)
                face_labels.append(label)

    features = np.array(detected_faces)
    features = features.reshape(features.shape[0], -1)

    return features, np.array(face_labels)
