"""In-process evaluator for the Heilbronn triangle problem (n=11)."""

import importlib.util
import itertools
import time
import traceback

import numpy as np

BENCHMARK = 0.036529889880030156
TOL = 1e-6
NUM_POINTS = 11

A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
UNIT_AREA = abs(
    A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1])
) / 2


def inside_triangle(points: np.ndarray) -> bool:
    sqrt3 = np.sqrt(3)
    for x, y in points:
        if not (y >= -TOL and sqrt3 * x <= sqrt3 - y + TOL and y <= sqrt3 * x + TOL):
            return False
    return True


def triangle_area(p1, p2, p3) -> float:
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2


def evaluate(program_path: str) -> dict:
    start = time.time()
    try:
        spec = importlib.util.spec_from_file_location("candidate_ht", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        points = module.heilbronn_triangle11()
        if not isinstance(points, np.ndarray):
            points = np.asarray(points)
        if points.shape != (NUM_POINTS, 2):
            return {"combined_score": 0.0, "error": f"bad shape {points.shape}"}
        if not inside_triangle(points):
            return {"combined_score": 0.0, "error": "point outside equilateral triangle"}

        min_area = min(
            triangle_area(p1, p2, p3)
            for p1, p2, p3 in itertools.combinations(points, 3)
        )
        min_area_normalized = float(min_area / UNIT_AREA)
        return {
            "min_area_normalized": min_area_normalized,
            "combined_score": float(min_area_normalized / BENCHMARK),
            "eval_time": float(time.time() - start),
        }
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "error": str(exc)[:300],
            "traceback": traceback.format_exc()[:1000],
        }
