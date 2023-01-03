import cv2
import numpy as np
from skimage import color, io, util


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


def color2gray(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:
        image = color.rgba2rgb(image)
    return util.img_as_float(color.rgb2gray(image))

def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("recognition/haarcascade_frontalface.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return image[y : y + w, x : x + h], input_img[y : y + w, x : x + h]