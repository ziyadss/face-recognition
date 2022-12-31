import os
from collections.abc import Iterator

import cv2
import numpy as np
from skimage import transform
from skimage.filters import gaussian

from common import utils

from .constants import OUTPUT_SHAPE

path = cv2.data.haarcascades
eye_detector = cv2.CascadeClassifier(os.path.join(path, "haarcascade_eye.xml"))


class Preprocessor:
    def euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def align(self, face: np.ndarray):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)
        eyes = eye_detector.detectMultiScale(face)

        if len(eyes) < 2:
            return face

        if len(eyes) > 2:
            eyes = sorted(eyes, key=lambda eye: eye[2], reverse=True)[:2]

        # Distinguish between left and right eye
        e1, e2 = eyes
        if e1[0] > e2[0]:
            left_eye = e2
            right_eye = e1
        else:
            left_eye = e1
            right_eye = e2

        # Right eye coordinates
        right_x = right_eye[0] + right_eye[2] // 2
        right_y = right_eye[1] + right_eye[3] // 2
        right_center = (right_x, right_y)
        # cv2.circle(face, right_center, 2, (255, 0, 0), 2)

        # Left eye coordinates
        left_x = left_eye[0] + left_eye[2] // 2
        left_y = left_eye[1] + left_eye[3] // 2
        left_center = (left_x, left_y)
        # cv2.circle(face, left_center, 2, (255, 0, 0), 2)

        # Find rotation direction
        if left_y > right_y:
            point_3rd = (right_x, left_y)
            direction = -1  # rotate CW
        else:
            point_3rd = (left_x, right_y)
            direction = 1  # rotate ACW

        a = self.euclidean_distance(left_center, point_3rd)
        b = self.euclidean_distance(right_center, point_3rd)
        c = self.euclidean_distance(right_center, left_center)

        if np.isclose(a, 0) or np.isclose(b, 0) or np.isclose(c, 0):
            return face
        cos_a = (b**2 + c**2 - a**2) / (2 * b * c)
        angle = np.degrees(np.arccos(cos_a))

        # angle = -direction * (90 - angle)
        if direction == -1:
            angle = 90 - angle

        return transform.rotate(face, direction * angle)

    def _singleRetinex(self, img: np.ndarray, sigma: int) -> np.ndarray:
        img[np.where(img == 0)] = 0.001
        retinex: np.ndarray = np.log10(img) - np.log10(gaussian(img, sigma))
    
        return retinex        
   
    def AMSR(self, image: np.ndarray, grayScaleImage: np.ndarray) -> np.ndarray:
     x, y, _  = image.shape
     mu_0, mu_1, mu_2, mu_3 = 32, 96, 160, 224
     sigma: int = 32
     
     R1_SSR: np.ndarray = self._singleRetinex(grayScaleImage, mu_1)
     R2_SSR: np.ndarray = self._singleRetinex(grayScaleImage, mu_2)
     R3_SSR: np.ndarray = self._singleRetinex(grayScaleImage, mu_3)
        
     R_images: list[np.ndarray] = [R1_SSR, R2_SSR, R3_SSR]
     # Calculate the luminence image
     Y: np.ndarray = np.zeros((x, y))
     for i in range(x):
         for j in range(y):
             Y[i][j] = 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]
             
     p_0, p_1, p_2, p_3 = np.ones((x, y)), np.zeros((x, y)), np.zeros((x, y)), np.ones((x, y))
     
     for i in range(x):
         for j in range(y):
             p_1[i][j] = np.exp(- (Y[i][j] - mu_1) / (2 * sigma ** 2)) # Small scale retinex SSR p
             p_2[i][j] = np.exp(- (Y[i][j] - mu_2) / (2 * sigma ** 2)) # Medium-scale retinex SSR p 
             p_3[i][j] = max(np.exp(- (Y[i][j] - mu_3) / (2 * sigma ** 2)), np.exp(- (Y[i][j] - mu_0) / (2 * sigma ** 2))) # Large scale retinex SSR p
             
     weights: np.ndarray = np.zeros((4, x, y))
     
     probabilities: list[np.ndarray] = [p_0, p_1, p_2, p_3]
     
     for s in range(4):
         for i in range(x):
             for j in range(y):
                 weights[s][i][j] = probabilities[s][i][j] / (probabilities[0][i][j] + probabilities[1][i][j] + probabilities[2][i][j])
     
     Y1_SSR: np.ndarray = np.zeros((x, y))
     Y2_SSR: np.ndarray = np.zeros((x, y))
     Y3_SSR: np.ndarray = np.zeros((x, y))
                 
     Y_images: list[np.ndarray] = [Y1_SSR, Y2_SSR, Y3_SSR]
     # Compute linear stretched images 
     for Y_image, R_image in zip(Y_images, R_images):
         percentile_99 = np.percentile(R_image, 99)
         percentile_1 = np.percentile(R_image, 1)
         for i in range(x):
             for j in range(y):
                 if R_image[i][j] > percentile_99:
                     Y_image[i][j] = 255
                 elif R_image[i][j] >= percentile_1 and R_image[i][j] <= percentile_99:
                     Y_image[i][j] = 255 * (R_image[i][j] - percentile_1) / (percentile_99 - percentile_1)
                 else:
                     Y_image[i][j] = 0 
                     
     Y_AMSR: np.ndarray = np.zeros((x, y))
     
     for i in range(x):
         for j in range(y):
             Y_AMSR[i][j] = weights[0][i][j] * Y[i][j] + (weights[1][i][j] * Y_images[0][i][j]) + (weights[2][i][j] * Y_images[1][i][j]) + (weights[3][i][j] * Y_images[2][i][j]) 
     
         
     return Y_AMSR / Y_AMSR.max() * 255 / 255.0

    def preprocess(
        self, images: list[np.ndarray] | Iterator[np.ndarray]
    ) -> list[np.ndarray]:
        # aligned = (self.align(image) for image in images)

        # grayed = (utils.color2gray(image) for image in aligned)

        ## grayed = (utils.color2gray(image) for image in images)
        ## 
        ## resized = (transform.resize(image, OUTPUT_SHAPE) for image in grayed)
        ## 
        ## flattened = (image.flatten() for image in resized)
        grayed = (utils.color2gray(image) for image in images)
        
        resized = (utils.color2gray(image) for image in grayed)

        resized_color = (utils.color2gray(image) for image in images)

        normalized = (self.AMSR(image, greyScaleImage) for image, greyScaleImage in zip(resized, resized_color))

        flattened = (image.flatten() for image in normalized)
        return list(flattened)
