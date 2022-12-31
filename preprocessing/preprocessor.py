import os
from collections.abc import Iterator

import cv2
import numpy as np
from skimage import transform

from common import utils

from .constants import OUTPUT_SHAPE

path = cv2.data.haarcascades
eye_detector = cv2.CascadeClassifier(os.path.join(path, "haarcascade_eye.xml"))


class Preprocessor:
    def euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def align(self, face: np.ndarray):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)
        eyes = eye_detector.detectMultiScale(face)

        if len(eyes) < 2:
            return face

        if len(eyes) > 2:
            eyes = sorted(eyes, key=lambda eye: eye[2], reverse=True)[:2]

        # Distinguish between left and right eye
        e1, e2 = eyes
        if e1[0] > e2[0]:
            left_eye = e2
            right_eye = e1
        else:
            left_eye = e1
            right_eye = e2

        # Right eye coordinates
        right_x = right_eye[0] + right_eye[2] // 2
        right_y = right_eye[1] + right_eye[3] // 2
        right_center = (right_x, right_y)
        # cv2.circle(face, right_center, 2, (255, 0, 0), 2)

        # Left eye coordinates
        left_x = left_eye[0] + left_eye[2] // 2
        left_y = left_eye[1] + left_eye[3] // 2
        left_center = (left_x, left_y)
        # cv2.circle(face, left_center, 2, (255, 0, 0), 2)

        # Find rotation direction
        if left_y > right_y:
            point_3rd = (right_x, left_y)
            direction = -1  # rotate CW
        else:
            point_3rd = (left_x, right_y)
            direction = 1  # rotate ACW

        a = self.euclidean_distance(left_center, point_3rd)
        b = self.euclidean_distance(right_center, point_3rd)
        c = self.euclidean_distance(right_center, left_center)

        if np.isclose(a, 0) or np.isclose(b, 0) or np.isclose(c, 0):
            return face
        cos_a = (b**2 + c**2 - a**2) / (2 * b * c)
        angle = np.degrees(np.arccos(cos_a))

        if direction == -1:
            angle = 90 - angle

        return transform.rotate(face, direction * angle)

    def preprocess(
        self, images: list[np.ndarray] | Iterator[np.ndarray]
    ) -> list[np.ndarray]:
        # aligned = (self.align(image) for image in images)

        # grayed = (utils.color2gray(image) for image in aligned)
        grayed = (utils.color2gray(image) for image in images)

        resized = (transform.resize(image, OUTPUT_SHAPE) for image in grayed)

        flattened = (image.flatten() for image in resized)

        return list(flattened)
