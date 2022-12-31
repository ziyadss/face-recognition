from collections.abc import Iterator

import numpy as np
from skimage import transform

from .constants import OUTPUT_SHAPE


class Preprocessor:
    def preprocess(
        self, images: list[np.ndarray] | Iterator[np.ndarray]
    ) -> list[np.ndarray]:
        resized = (transform.resize(image, OUTPUT_SHAPE) for image in images)

        flattened = (image.flatten() for image in resized)

        return list(flattened)
