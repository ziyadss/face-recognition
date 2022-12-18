import numpy as np
from scipy import signal

from constants import *


def get_histogram(magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    bin1 = (orientation // BIN_WIDTH).astype(int)
    bin2 = (bin1 + 1) % NBINS
    bin1_val = bin1 * BIN_WIDTH

    overflow = magnitude * (orientation - bin1_val) / BIN_WIDTH
    non_overflow = magnitude - overflow
    
    bins = np.bincount(bin1, weights=non_overflow, minlength=NBINS) + np.bincount(bin2, weights=overflow, minlength=NBINS)
    # bincount is just `for i in range(len(x)): bins[x[i]] += weights[i]`
    # but np's implementation is in C, giving a 40% speedup than implementing it in python

    return bins # type: ignore

def normalize_block(block: np.ndarray) -> np.ndarray:
    # Scores below are for LinearSVC. SVC scores are slightly better.
    # Might still need to recalculate for SVC to determine best norm.
    #  SVC scores for 2pi:
    #  L1-sqrt norm: 0.9944403261675315

    # # L1 norm
    # # pi : 0.9799851742031134
    # # 2pi: 0.9870274277242401
    # norm = np.linalg.norm(block, ord=1)
    # if norm == 0:
    #     return block
    # return block / norm

    # # L2 norm
    # # pi : 0.9822090437361009
    # # 2pi: 0.9892512972572276
    # norm = np.linalg.norm(block, ord=2)
    # if norm == 0:
    #     return block
    # return block / norm

    # L1-sqrt norm
    # pi : 0.9877687175685693
    # 2pi: 0.9929577464788732
    norm = np.linalg.norm(block, ord=1)
    if norm == 0:
        return block
    return np.sqrt(block / norm)

    # # L2-sqrt norm
    # # pi : 0.9844329132690882
    # # 2pi: 0.991475166790215
    # norm = np.linalg.norm(block, ord=2)
    # if norm == 0:
    #     return block
    # return np.sqrt(block / norm)

    # # L2-hys norm
    # # pi : 0.9873980726464048
    # # 2pi: 0.9903632320237212
    # norm = np.linalg.norm(block, ord=2)
    # if norm == 0:
    #     return block
    # block = np.clip(block / norm, 0, 0.2)
    # norm = np.linalg.norm(block, ord=2)
    # return block / norm

X_FILTER = np.array([[-1, 0, 1]])
Y_FILTER = X_FILTER.T

def hog(image: np.ndarray) -> np.ndarray:
    x: np.ndarray = signal.convolve2d(image, X_FILTER, mode="same")
    y: np.ndarray = signal.convolve2d(image, Y_FILTER, mode="same")

    magnitude = np.sqrt(x ** 2 + y ** 2)
    orientation = np.arctan2(y, x)

    # Lineary scale orientation to [0, ORIENTATION_RANGE)
    orientation = (ORIENTATION_SCALE_M * orientation + ORIENTATION_SCALE_C) % ORIENTATION_RANGE

    h, w = image.shape
    H, W = PIXELS_PER_CELL
    histograms = []
    for i in range(0, h, H):
        histograms.append([])
        for j in range(0, w, W):
            mag = magnitude[i : i + H, j : j + W].flatten()
            ori = orientation[i : i + H, j : j + W].flatten()
            cell = get_histogram(mag, ori)
            histograms[-1].append(cell)

    histograms = np.array(histograms)

    blocks = []
    H, W = CELLS_PER_BLOCK
    for i in range(histograms.shape[0] - H + 1):
        blocks.append([])
        for j in range(histograms.shape[1] - W + 1):
            block = histograms[i : i + H, j : j + W].flatten()
            block = normalize_block(block)
            blocks[-1].append(block)

    blocks = np.array(blocks)

    return blocks.flatten()
