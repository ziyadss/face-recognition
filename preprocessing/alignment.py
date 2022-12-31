import os
import cv2
from skimage import transform
import pandas as pd
import numpy as np

path = cv2.data.haarcascades
eye_detector = cv2.CascadeClassifier(os.path.join(path, "haarcascade_eye.xml"))

test_set = ["data/one.jpg", "data/two.png"]

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def alignment(face):
    eyes = eye_detector.detectMultiScale(face)

    if len(eyes) < 2:
        raise Exception("No eyes detected")

    if len(eyes) > 2:
        eye = eyes[:, 2]
        container1 = []
        for i in range(0, len(eye)):
            container = (eye[i], i)
            container1.append(container)
        df = pd.DataFrame(container1, columns=["length", "idx"]).sort_values(by=['length'])
        
        eyes = eyes[df.idx.values[0:2]]

    # choose between left or right eye
    e1, e2 = eyes
    if e1[0] > e2[0]:
        left_eye = e2
        right_eye = e1
    else:
        left_eye = e1
        right_eye = e2

    # center of eyes

    # center of right eye
    right_eye_center = (right_eye[0] + right_eye[2] //2 , right_eye[1] + right_eye[3] // 2)
    right_eye_x, right_eye_y = right_eye_center


    # center of left eye
    left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
    left_eye_x, left_eye_y = left_eye_center

    # finding rotation direction
    if left_eye_y > right_eye_y:
        print("Rotate CW")
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 # rotate CW
    else:
        print("Rotate ACW")
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 # rotate ACW

    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, point_3rd)
    c = euclidean_distance(right_eye_center, left_eye_center)

    cos_a = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    angle = np.degrees(np.arccos(cos_a))
    # angle = -direction * (90 - angle)
    print("Rotation angle: ", angle)

    return transform.rotate(face, direction * angle)

