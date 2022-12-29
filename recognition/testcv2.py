import cv2
import numpy as np
from sklearn.metrics import accuracy_score

from constants import TESTING_PATH, TRAINING_PATH
from helpers import gaborize, prepare_data

detected_faces, face_labels = prepare_data(TRAINING_PATH)

detected_faces, face_labels = np.array(detected_faces), np.array(face_labels)
detected_faces = detected_faces.reshape(detected_faces.shape[0], -1)

print("Total faces: ", len(detected_faces))
print("Total labels: ", len(face_labels))

print(detected_faces.shape)

print("Training Fisherfaces...")
fisher = cv2.face.FisherFaceRecognizer_create()
fisher.train(detected_faces, face_labels)
print("Fisherfaces trained.")

print("Training Gaborized Fisherfaces...")
fisher_gaborized = cv2.face.FisherFaceRecognizer_create()
fisher_gaborized.train(gaborize(detected_faces), face_labels)
print("Gaborized Fisherfaces trained.")

print("Predicting With Fisherfaces...")
predicted_lables = []
faces, lables = prepare_data(TESTING_PATH)
faces, lables = np.array(faces), np.array(lables)
faces = faces.reshape(faces.shape[0], -1)
print(faces.shape)

for face in faces:
    lable = fisher.predict(face)
    predicted_lables.append(lable[0])

print("Predicting With Gaborized Fisherfaces...")
predicted_lables_gaborized = []
faces, lables = prepare_data(TESTING_PATH)
faces, lables = np.array(faces), np.array(lables)
faces = faces.reshape(faces.shape[0], -1)
print(faces.shape)

for face in gaborize(faces):
    lable = fisher_gaborized.predict(face)
    predicted_lables_gaborized.append(lable[0])

print(lables)
print(predicted_lables)
print(predicted_lables_gaborized)

print(accuracy_score(lables, predicted_lables))
print(accuracy_score(lables, predicted_lables_gaborized))
