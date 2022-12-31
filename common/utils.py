import numpy as np
from skimage import color, io, util


def read_as_float(path: str) -> np.ndarray:
    return util.img_as_float(io.imread(path, as_gray=True))


def color2gray(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:
        image = color.rgba2rgb(image)
    return util.img_as_float(color.rgb2gray(image))
