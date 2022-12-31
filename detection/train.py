import csv
from time import perf_counter_ns

import numpy as np
from skimage import io, transform

from common import utils

from .constants import WINDOW_SHAPE
from .detector import FaceDetector

NON_FACES_COUNT = 1
TRAIN_TO_TEST_RATIO = 1
IMAGES_DIR = "C:/Users/ziyad/Downloads/celebs/img_celeba"

i = 0
training_data = []
testing_data = []
with open("data/filtered_information.csv", "r") as fd:
    reader = csv.reader(fd)
    next(reader)

    for row in reader:
        i += 1
        if i % TRAIN_TO_TEST_RATIO != 0:
            training_data.append(row)
        else:
            testing_data.append(row)


def process_data(data):
    colored_faces = []
    faces = []
    non_faces = []
    labels = []
    for row in data:
        file = row[0]
        x1, y1 = int(row[1]), int(row[2])
        w, h = int(row[3]), int(row[4])

        colored = io.imread(f"{IMAGES_DIR}/{file}")
        gray = utils.color2gray(colored)
        H, W = gray.shape

        face = transform.resize(gray[y1 : y1 + h, x1 : x1 + w], WINDOW_SHAPE)
        faces.append(face)

        # get random boxes that are not the face
        HW, WW = WINDOW_SHAPE
        count = 0
        while count < NON_FACES_COUNT:
            x = np.random.randint(0, W - WW)
            y = np.random.randint(0, H - HW)
            if x >= x1 and x <= x1 + w and y >= y1 and y <= y1 + h:
                continue
            count += 1
            non_face = gray[y : y + HW, x : x + WW]
            non_faces.append(non_face)

    return colored_faces, faces, non_faces, labels


# Modules
detector = FaceDetector()

# Train
start = perf_counter_ns()
colored, faces, non_faces, labels = process_data(training_data)
end = perf_counter_ns()
print(
    f"Training faces: {len(faces)}, non-faces: {len(non_faces)}, time: {(end - start) / 1e9} seconds"
)

start = perf_counter_ns()
detector.train(faces, non_faces)
detector.dump()
end = perf_counter_ns()
print(f"Detector training time: {(end - start) / 1e9} seconds")

# Test
start = perf_counter_ns()
colored, faces, non_faces, labels = process_data(testing_data)
end = perf_counter_ns()
print(
    f"Testing faces: {len(faces)}, non-faces: {len(non_faces)}, time: {(end - start) / 1e9} seconds"
)

detector_score = detector.test(faces, non_faces)

print(f"Detector score: {detector_score}")
