import numpy as np

## Model constants
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
NBINS = 9

ORIENTATION_RANGE = 2 * np.pi
BIN_WIDTH = ORIENTATION_RANGE / NBINS

ORIENTATION_SCALE_C = ORIENTATION_RANGE / 2
ORIENTATION_SCALE_M = ORIENTATION_SCALE_C / np.pi

FACE = 1
NON_FACE = -1

TRAIN_FACES_DIR = "detection/images/train/positive"
TRAIN_NON_FACES_DIR = "detection/images/train/negative"

TEST_FACES_DIR = "detection/images/test/positive"
TEST_NON_FACES_DIR = "detection/images/test/negative"

CLASSIFIER_PATH = "detection/classifier.pkl"

## Detection constants
PROCESSES_COUNT = 4

BINARY_THRESHOLD = 0
NMS_THRESHOLD = 0.3

WINDOW_SHAPE = (37, 37)
WINDOW_SHIFT = (5, 5)
