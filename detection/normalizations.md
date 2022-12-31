```py
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

    # # L2-hys norm
    # # pi : 0.9873980726464048
    # # 2pi: 0.9903632320237212
    # norm = np.linalg.norm(block, ord=2)
    # if norm == 0:
    #     return block
    # block = np.clip(block / norm, 0, 0.2)
    # norm = np.linalg.norm(block, ord=2)
    # return block / norm
```