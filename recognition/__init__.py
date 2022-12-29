import cv2
import fisher
import numpy as np
from constants import TESTING_PATH, TRAINING_PATH
from helpers import gaborize, prepare_data
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

detected_faces, face_labels = prepare_data(TRAINING_PATH)

detected_faces, face_labels = np.array(detected_faces), np.array(face_labels)
detected_faces = detected_faces.reshape(detected_faces.shape[0], -1)

print("Total faces: ", len(detected_faces))
print("Total labels: ", len(face_labels))

print(detected_faces.shape)

print("Training Fisherfaces...")
fisherfaces = fisher.fisherfaces2()
fisherfaces.fit(detected_faces, face_labels)
svm = SVC()
svm.fit(fisherfaces.project(detected_faces), face_labels)
# svm.fit(fisherfaces.fit_transform(detected_faces, face_labels), face_labels)
print("Fisherfaces trained.")

print("Predicting With Fisherfaces...")
faces, lables = prepare_data(TESTING_PATH)
faces, lables = np.array(faces), np.array(lables)
faces = faces.reshape(faces.shape[0], -1)
print(faces.shape)
newf = fisher.fisherfaces()
predicted_lables = svm.predict(fisherfaces.project(faces))
# predicted_lables = svm.predict(newf.fit_transform(faces, lables))
# predicted_lables = svm.predict(lca.transform(fisher.pca(faces)))

print(lables)
print(predicted_lables)
print(accuracy_score(lables, predicted_lables))

# print("Training Gaborized Fisherfaces...")
# fisherfaces_gaborized = fisher.fisherfaces(gaborize(detected_faces), face_labels)
# svm_gaborized = SVC()
# svm_gaborized.fit(fisherfaces_gaborized, face_labels)
# print("Gaborized Fisherfaces trained.")

# print("Predicting With Gaborized Fisherfaces...")
# faces, lables = prepare_data(TESTING_PATH)
# faces, lables = np.array(faces), np.array(lables)
# faces = faces.reshape(faces.shape[0], -1)
# print(faces.shape)
# predicted_lables_gaborized = svm.predict(fisher.fisherfaces(gaborize(faces), lables))

# print(lables)
# print(predicted_lables_gaborized)
# print(accuracy_score(lables, predicted_lables_gaborized))