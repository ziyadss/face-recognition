import os
from typing import List

import cv2
import numpy as np
from constants import CLASSIFIER_PATH


def asColumns(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.array([])
    v = np.empty((X[0].size, len(X)), dtype=X[0].dtype)
    for i, x in enumerate(X):
        v[:, i] = x.flatten()
    return v


def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(CLASSIFIER_PATH)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return image[y : y + w, x : x + h], faces[0]


def prepare_data(path):
    detected_faces = []
    face_labels = []
    image_dirs = os.listdir(path)
    for dir_name in image_dirs:
        label = int(dir_name)
        images_path = path + "/" + dir_name
        images_names = os.listdir(images_path)

        for image_name in images_names:
            image_path = images_path + "/" + image_name
            image = cv2.imread(image_path)
            face, _ = detect_face(image)
            if face is not None:
                resized_face = cv2.resize(face, (21, 21), interpolation=cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(label)

    return detected_faces, face_labels


def gaborize(images: np.ndarray) -> np.ndarray:
    result = []
    kernels = []
    for theta in range(4):
        theta = theta / 4.0 * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 2):
                for gamma in (0.05, 0.5):
                    kernel = cv2.getGaborKernel(
                        (21, 21), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F
                    )
                    kernels.append(kernel)
    for image in images:
        d = image.shape[0]
        if len(image.shape) > 1:
            d *= image.shape[1]
        gaborized = np.zeros(d * (len(kernels) + 1))
        gaborized[0:d] = image.flatten()
        for i, kernel in enumerate(kernels):
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            gaborized[d * (i + 1) : d * (i + 2)] = abs(filtered.flatten())
        result.append(gaborized)
    return np.array(result)


def histogram_equalization(X: np.ndarray) -> np.ndarray:
    # Convert the input array to a 3D array if it is not already
    if len(X.shape) == 2:
        X = X[:, :, np.newaxis]
    elif len(X.shape) == 3 and X.shape[2] == 1:
        X = X[:, :, 0]

    # Perform histogram equalization on each channel of the input array
    X_equalized = np.zeros(X.shape)
    for i in range(X.shape[2]):
        # Convert the channel to 8-bit single channel image before calling cv2.equalizeHist
        X_8uc1 = X[:, :, i].astype(np.uint8)
        X_equalized[:, :, i] = cv2.equalizeHist(X_8uc1)

    # Convert the output array back to a 2D array if it is 3D with only one channel
    if len(X_equalized.shape) == 3 and X_equalized.shape[2] == 1:
        X_equalized = X_equalized[:, :, 0]

    return X_equalized
