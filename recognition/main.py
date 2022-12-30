import numpy as np
from sklearn.metrics import accuracy_score

from recognition.constants import TESTING_PATH, TRAINING_PATH
from recognition.fisher import FisherRecognizer
from recognition.helpers import prepare_data

n = 40
l = 5


detected_faces, face_labels = prepare_data(TRAINING_PATH, n, l)
detected_faces, face_labels = np.array(detected_faces), np.array(face_labels)
detected_faces = detected_faces.flatten()
# detected_faces = gaborize(detected_faces)

print(detected_faces.shape)

print("Total faces: ", len(detected_faces))
print("Total labels: ", len(face_labels))


fisher = FisherRecognizer()
fisher.fit(np.array(detected_faces), np.array(face_labels))

faces, lables = prepare_data(TESTING_PATH, n)
faces, lables = np.array(faces), np.array(lables)
faces = faces.reshape(faces.shape[0], -1)
# faces = gaborize(faces)

print(faces.shape)

predicted_lables = fisher.predict(faces)


print(lables)
print(predicted_lables)
print(accuracy_score(lables, predicted_lables))
