"""Initial constructor for n=26 circle packing in the unit square.

Target: AlphaEvolve sum_radii = 2.635 for n=26.
``construct_packing()`` must return ``(centers, radii, sum_radii)``.
"""

import numpy as np


def construct_packing():
    n = 26
    centers = np.zeros((n, 2))

    centers[0] = [0.5, 0.5]
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]
    centers = np.clip(centers, 0.01, 0.99)

    radii = compute_max_radii(centers)
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    n = centers.shape[0]
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii


def run_packing():
    return construct_packing()
