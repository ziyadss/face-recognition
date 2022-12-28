import pickle

import numpy as np
from skimage import io
from sklearn import svm

from constants import CLASSIFIER_PATH
from detection import detect_with_scales, BoxType
from helpers import read_as_float

with open(CLASSIFIER_PATH, "rb") as fd:
    clf: svm.SVC = pickle.load(fd)

path: str = "images/pic.jpg"
scales: list[float] = [1]
# a smaller scale gets bigger faces

img: np.ndarray = read_as_float(path)
faces: list[BoxType] = detect_with_scales(clf, img, scales)

img_original: np.ndarray = io.imread(path)
channels = img_original.shape[2]
if channels == 3:
    for x1, y1, x2, y2, score, scale in faces:
        img_original[x1, y1:y2] = [255, 0, 0]
        img_original[x1:x2, y1] = [255, 0, 0]
        img_original[x2, y1:y2] = [255, 0, 0]
        img_original[x1:x2, y2] = [255, 0, 0]
elif channels == 4:
    for x1, y1, x2, y2, score, scale in faces:
        img_original[x1, y1:y2, 0:3] = [255, 0, 0]
        img_original[x1:x2, y1, 0:3] = [255, 0, 0]
        img_original[x2, y1:y2, 0:3] = [255, 0, 0]
        img_original[x1:x2, y2, 0:3] = [255, 0, 0]
else:
    raise ValueError("Invalid number of channels")

io.imshow(img_original)
io.show()
