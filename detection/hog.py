import numpy as np
from scipy import signal

from constants import *


def get_histogram(magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    bin1 = (orientation // BIN_WIDTH).astype(int)
    bin2 = (bin1 + 1) % NBINS
    bin1_val = bin1 * BIN_WIDTH

    overflow = magnitude * (orientation - bin1_val) / BIN_WIDTH
    non_overflow = magnitude - overflow

    bins = np.bincount(bin1, weights=non_overflow, minlength=NBINS) + np.bincount(
        bin2, weights=overflow, minlength=NBINS
    )
    # bincount is just `for i in range(len(x)): bins[x[i]] += weights[i]`
    # but np's implementation is in C, giving a 40% speedup than implementing it in python

    return bins  # type: ignore


X_FILTER = np.array([[-1, 0, 1]])
Y_FILTER = X_FILTER.T


def hog(image: np.ndarray) -> np.ndarray:
    x: np.ndarray = signal.convolve2d(image, X_FILTER, mode="same")
    y: np.ndarray = signal.convolve2d(image, Y_FILTER, mode="same")

    magnitude = np.sqrt(x**2 + y**2)
    orientation = np.arctan2(y, x)

    # Lineary scale orientation to [0, ORIENTATION_RANGE)
    orientation = (ORIENTATION_SCALE_M * orientation + ORIENTATION_SCALE_C) % ORIENTATION_RANGE

    h, w = image.shape
    H, W = PIXELS_PER_CELL
    histograms = []
    for i in range(0, h, H):
        to_add = []
        for j in range(0, w, W):
            mag = magnitude[i : i + H, j : j + W].flatten()
            ori = orientation[i : i + H, j : j + W].flatten()
            cell = get_histogram(mag, ori)
            to_add.append(cell)
        histograms.append(to_add)

    histograms = np.array(histograms)

    h, w, *_ = histograms.shape
    H, W = CELLS_PER_BLOCK
    blocks = []
    for i in range(h - H + 1):
        to_add = []
        for j in range(w - W + 1):
            block = histograms[i : i + H, j : j + W].flatten()
            blocks.append(block)
        blocks.append(to_add)

    blocks = np.array(blocks)

    norms = np.linalg.norm(blocks, axis=2, keepdims=True, ord=1)
    np.divide(blocks, norms, out=blocks, where=norms != 0)
    np.sqrt(blocks, out=blocks)

    return blocks.flatten()
