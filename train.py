import csv

import numpy as np
from skimage import transform
import cv2

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
    faces = []
    non_faces = []
    labels = []
    for row in data:
        file = row[0]
        x1, y1 = int(row[1]), int(row[2])
        w, h = int(row[3]), int(row[4])
        identity = int(row[5])

        image = utils.read_as_float(f"{IMAGES_DIR}/{file}")
        H, W = image.shape

        face = transform.resize(
            image[y1 : y1 + h, x1 : x1 + w], DConstants.WINDOW_SHAPE
        )
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
            non_face = image[y : y + HW, x : x + WW]
            non_faces.append(non_face)

    return faces, non_faces, labels


# Modules
detector = FaceDetector()
preprocessor = Preprocessor()
recognizer = FisherRecognizer()


# Train
faces, non_faces, labels = process_data(training_data)
print(f"Training faces: {len(faces)}, non-faces: {len(non_faces)}")

detector.train(faces, non_faces)
detector.dump()

preprocessed_faces = preprocessor.preprocess(faces)

recognizer.fit(preprocessed_faces, labels)
recognizer.dump()

recog = cv2.face.FisherFaceRecognizer_create()
recog.train(np.array(preprocessed_faces), np.array(labels))

# Test
detector.load()
recognizer.load()
faces, non_faces, labels = process_data(testing_data)
print(f"Testing faces: {len(faces)}, non-faces: {len(non_faces)}")

detector_score = detector.test(faces, non_faces)
preprocessed_faces = preprocessor.preprocess(faces)
recognizer_score = recognizer.score(preprocessed_faces, labels)

print(f"Detector score: {detector_score}")
print(f"Recognizer score: {recognizer_score}")

preds = recognizer.predict(preprocessed_faces)
count = sum(1 for i in range(len(labels)) if labels[i] == preds[i])
print(f"Predictions: {count}/{len(labels)}")

# cv2 face recognition
preds = [recog.predict(f)[0] for f in preprocessed_faces]
count = sum(1 for i in range(len(labels)) if labels[i] == preds[i])
print(f"Predictions: {count}/{len(labels)}")