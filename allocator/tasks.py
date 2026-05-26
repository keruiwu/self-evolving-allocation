"""Task registry: initial program + evaluator + system prompt for each problem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TASKS_ROOT = Path(__file__).resolve().parent.parent / "tasks"


CP_SYSTEM = """\
You are an expert mathematician specializing in circle packing problems and computational geometry.
Your task is to improve a constructor function that places 26 circles in a unit square to maximize the sum of their radii.

Target: AlphaEvolve achieved sum of radii = 2.635 for n=26.
- The objective is to maximize sum(radii) subject to no-overlap and unit-square containment.
- Good initial placements (hexagonal patterns, corner utilization, edge circles) matter.
- The function `construct_packing()` must return (centers, radii, sum_radii).
"""

MMD_SYSTEM = """\
You are an expert computational geometer focused on point-dispersion problems.
Your task is to improve a constructor that produces 16 points in 2D maximizing (d_min / d_max)^2.

- Target: beat the AlphaEvolve benchmark 1 / 12.889266112 ≈ 0.0776.
- combined_score = (d_min / d_max)^2 / BENCHMARK.
- The function `min_max_dist_dim2_16()` must return a numpy array of shape (16, 2).
"""

HT_SYSTEM = """\
You are an expert computational geometer focused on the Heilbronn triangle problem.
Your task is to improve a constructor that places 11 points inside the equilateral triangle with
vertices (0,0), (1,0), (0.5, sqrt(3)/2), maximizing the area of the smallest triangle formed by any
three of the placed points.

- Target benchmark: 0.036529889880030156. combined_score = min_area_normalized / BENCHMARK.
- The function `heilbronn_triangle11()` must return a numpy array of shape (11, 2).
- All 11 points must lie inside the equilateral triangle (within tolerance 1e-6).
"""


@dataclass(frozen=True)
class TaskSpec:
    key: str
    initial_program: Path
    evaluator: Path
    system_message: str
    eval_timeout_s: int = 60


TASKS: dict[str, TaskSpec] = {
    "cp": TaskSpec(
        key="cp",
        initial_program=TASKS_ROOT / "circle_packing" / "initial_program.py",
        evaluator=TASKS_ROOT / "circle_packing" / "evaluator.py",
        system_message=CP_SYSTEM,
        eval_timeout_s=120,
    ),
    "mmd": TaskSpec(
        key="mmd",
        initial_program=TASKS_ROOT / "min_max_dist" / "initial_program.py",
        evaluator=TASKS_ROOT / "min_max_dist" / "evaluator.py",
        system_message=MMD_SYSTEM,
        eval_timeout_s=60,
    ),
    "ht": TaskSpec(
        key="ht",
        initial_program=TASKS_ROOT / "heilbronn_triangle" / "initial_program.py",
        evaluator=TASKS_ROOT / "heilbronn_triangle" / "evaluator.py",
        system_message=HT_SYSTEM,
        eval_timeout_s=60,
    ),
}


def get_task(key: str) -> TaskSpec:
    if key not in TASKS:
        raise KeyError(f"Unknown task {key!r}. Available: {sorted(TASKS)}")
    return TASKS[key]
