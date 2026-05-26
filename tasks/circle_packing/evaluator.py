"""In-process evaluator for n=26 circle packing.

The orchestrator invokes ``evaluate(program_path)`` via a subprocess, so it is safe to
run heavy / unsafe candidate code in-process here.
"""

import importlib.util
import time
import traceback

import numpy as np

TARGET_VALUE = 2.635
N = 26


def validate_packing(centers: np.ndarray, radii: np.ndarray) -> bool:
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    if (radii < 0).any():
        return False
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if dist + 1e-6 < radii[i] + radii[j]:
                return False
    return True


def evaluate(program_path: str) -> dict:
    start = time.time()
    try:
        spec = importlib.util.spec_from_file_location("candidate_cp", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "run_packing"):
            centers, radii, reported_sum = module.run_packing()
        else:
            centers, radii, reported_sum = module.construct_packing()

        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float)
        if centers.shape != (N, 2) or radii.shape != (N,):
            return {"combined_score": 0.0, "error": f"bad shapes {centers.shape}/{radii.shape}"}

        valid = validate_packing(centers, radii)
        sum_radii = float(np.sum(radii)) if valid else 0.0
        target_ratio = sum_radii / TARGET_VALUE if valid else 0.0
        return {
            "sum_radii": sum_radii,
            "target_ratio": target_ratio,
            "validity": 1.0 if valid else 0.0,
            "combined_score": float(target_ratio),
            "eval_time": float(time.time() - start),
        }
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "error": str(exc)[:300],
            "traceback": traceback.format_exc()[:1000],
        }
