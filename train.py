import csv
from time import perf_counter_ns

import cv2
import numpy as np
from skimage import io, transform

from common import utils
from detection import constants as DConstants
from detection.detector import FaceDetector
from preprocessing.preprocessor import Preprocessor
from recognition.fisher import FisherRecognizer

NON_FACES_COUNT = 1
TRAIN_TO_TEST_RATIO = 5
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
        identity = int(row[5])

        colored = io.imread(f"{IMAGES_DIR}/{file}")
        gray = utils.color2gray(colored)
        H, W = gray.shape

        colored_face = colored[y1 : y1 + h, x1 : x1 + w]
        face = transform.resize(gray[y1 : y1 + h, x1 : x1 + w], DConstants.WINDOW_SHAPE)
        colored_faces.append(colored_face)
        faces.append(face)
        labels.append(identity)

        # get random boxes that are not the face
        HW, WW = DConstants.WINDOW_SHAPE
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
preprocessor = Preprocessor()
recognizer = FisherRecognizer()
cv2_recognizer = cv2.face.EigenFaceRecognizer_create()

# Train
start = perf_counter_ns()
colored, faces, non_faces, labels = process_data(training_data)
end = perf_counter_ns()
print(
    f"Training faces: {len(faces)}, non-faces: {len(non_faces)}, time: {(end - start) / 1e9} seconds"
)

# start = perf_counter_ns()
# detector.train(faces, non_faces)
# detector.dump()
# end = perf_counter_ns()
# print(f"Detector training time: {(end - start) / 1e9} seconds")

start = perf_counter_ns()
preprocessed_faces = preprocessor.preprocess(colored)
end = perf_counter_ns()
print(f"Preprocessing time: {(end - start) / 1e9} seconds")

start = perf_counter_ns()
recognizer.fit(preprocessed_faces, labels)
recognizer.dump()
end = perf_counter_ns()
print(f"Recognizer training time: {(end - start) / 1e9} seconds")

start = perf_counter_ns()
cv2_recognizer.train(np.array(preprocessed_faces), np.array(labels))
end = perf_counter_ns()
print(f"OpenCV Recognizer training time: {(end - start) / 1e9} seconds")

# Test
start = perf_counter_ns()
colored, faces, non_faces, labels = process_data(testing_data)
end = perf_counter_ns()
print(
    f"Testing faces: {len(faces)}, non-faces: {len(non_faces)}, time: {(end - start) / 1e9} seconds"
)

detector.load()
recognizer.load()

detector_score = detector.test(faces, non_faces)
preprocessed_faces = preprocessor.preprocess(colored)
recognizer_score = recognizer.score(preprocessed_faces, labels)

print(f"Detector score: {detector_score}")
print(f"Recognizer score: {recognizer_score}")

preds = recognizer.predict(preprocessed_faces)
count = sum(1 for i in range(len(labels)) if labels[i] == preds[i])
print(f"Predictions: {count}/{len(labels)}")

# cv2 face recognition
preds = [cv2_recognizer.predict(f)[0] for f in preprocessed_faces]
count = sum(1 for i in range(len(labels)) if labels[i] == preds[i])
print(f"Predictions: {count}/{len(labels)}")
