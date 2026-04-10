import numpy as np


def init_state():
    return {
        "P": np.empty((2, 0), dtype=np.float32),
        "X": np.empty((3, 0), dtype=np.float32),
        "C": np.empty((2, 0), dtype=np.float32),
        "F": np.empty((2, 0), dtype=np.float32),
        "T_first": np.empty((0, 4, 4), dtype=np.float64),
        "T_wc": np.eye(4, dtype=np.float64),
        "K": None,
    }
