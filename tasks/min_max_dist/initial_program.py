"""Initial constructor for the min/max-distance ratio problem (16 points in 2D)."""

import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, 2))
