"""In-process evaluator for the min/max distance ratio task (16 points, 2D)."""

import importlib.util
import time
import traceback

import numpy as np
import scipy.spatial.distance as sp_dist

BENCHMARK = 1 / 12.889266112
NUM_POINTS = 16
DIM = 2


def evaluate(program_path: str) -> dict:
    start = time.time()
    try:
        spec = importlib.util.spec_from_file_location("candidate_mmd", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        points = module.min_max_dist_dim2_16()
        if not isinstance(points, np.ndarray):
            points = np.asarray(points)
        if points.shape != (NUM_POINTS, DIM):
            return {"combined_score": 0.0, "error": f"bad shape {points.shape}"}

        pairwise = sp_dist.pdist(points)
        d_min = float(np.min(pairwise))
        d_max = float(np.max(pairwise))
        inv_ratio_sq = (d_min / d_max) ** 2 if d_max > 0 else 0.0
        return {
            "min_max_ratio": float(inv_ratio_sq),
            "combined_score": float(inv_ratio_sq / BENCHMARK),
            "eval_time": float(time.time() - start),
        }
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "error": str(exc)[:300],
            "traceback": traceback.format_exc()[:1000],
        }
