import pickle
from time import perf_counter_ns

from sklearn import metrics, svm

from .constants import *
from .helpers import directory_hogs

print("Getting train vectors\n")
start = perf_counter_ns()

faces_hog = directory_hogs(TRAIN_FACES_DIR)
non_faces_hog = directory_hogs(TRAIN_NON_FACES_DIR)

end = perf_counter_ns()
print(f"Time: {(end - start) / 1e9} seconds")

train_data = faces_hog + non_faces_hog
train_labels = [FACE] * len(faces_hog) + [NON_FACE] * len(non_faces_hog)

print(f"\nTraining on {len(train_data)} images")
clf = svm.SVC()
clf.fit(train_data, train_labels)

print("\nSaving classifier")
with open(CLASSIFIER_PATH, "wb") as fd:
    pickle.dump(clf, fd)

print(f"\nTesting on training images")
train_score = clf.score(train_data, train_labels)
print(f"Score: {train_score}")


print("\nGetting test vectors")
faces_hog = directory_hogs(TEST_FACES_DIR)
non_faces_hog = directory_hogs(TEST_NON_FACES_DIR)

test_data = faces_hog + non_faces_hog
test_labels = [FACE] * len(faces_hog) + [NON_FACE] * len(non_faces_hog)

print(f"\nTesting on {len(test_data)} images")
test_score = clf.score(test_data, test_labels)
print(f"Score: {test_score}")


def predict(vec):
    score = clf.decision_function(vec)
    idx = (score > BINARY_THRESHOLD).astype(int)
    return clf.classes_[idx]


new_score = metrics.accuracy_score(test_labels, predict(test_data))
print(f"Thresholded score: {new_score}")
