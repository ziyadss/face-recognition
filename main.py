import pickle

import numpy as np
from skimage import io
from sklearn import svm

from constants import CLASSIFIER_PATH
from detection import detect_with_scales, face_type
from helpers import read_as_float

with open(CLASSIFIER_PATH, "rb") as fd:
    clf: svm.SVC = pickle.load(fd)

path: str = "images/pic.jpg"
scales: list[float] = [1]
# a smaller scale gets bigger faces

img: np.ndarray = read_as_float(path)
faces: list[face_type] = detect_with_scales(clf, img, scales)

img_original: np.ndarray = io.imread(path)
for x1, y1, x2, y2, score, scale in faces:
    img_original[x1, y1:y2] = [255, 0, 0]
    img_original[x1:x2, y1] = [255, 0, 0]
    img_original[x2, y1:y2] = [255, 0, 0]
    img_original[x1:x2, y2] = [255, 0, 0]

io.imshow(img_original)
io.show()
