import numpy as np
from skimage import transform


class PreprocessorConfiguration:
    def __init__(self, output_shape: tuple[int, int]):
        self.output_shape = output_shape


class Preprocessor:
    def __init__(self, config: PreprocessorConfiguration):
        self.config = config

    def preprocess(self, images: list[np.ndarray]) -> list[np.ndarray]:
        resized = [
            transform.resize(image, self.config.output_shape) for image in images
        ]

        return resized
