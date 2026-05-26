"""Greedy / clean Φ-model evolution.

A *generation* freezes the current best, asks the LLM for N independent rewrites in
parallel, evaluates them, and accepts the best child if it beats the parent. Total
budget per run is ``C = T * N`` LLM calls.

``GreedyRunner`` is stateful so the bandit orchestrator can call ``step_generation``
one round at a time on each arm. ``run_all_generations`` is the one-shot helper used
by the standalone CLI.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Any

from .evaluator import evaluate_code
from .flops import flops_per_round, flops_summary
from .llm import LLMClient
from .parser import parse_full_rewrite
from .prompt import build_user_message
from .tasks import TaskSpec

logger = logging.getLogger(__name__)


def _child_seed(run_seed: int, N: int, gen: int, idx: int) -> int:
    """Deterministic per-(run, N, gen, child) seed (matches openevolve convention)."""
    data = struct.pack(">IIII", run_seed & 0xFFFFFFFF, N & 0xFFFFFFFF, gen & 0xFFFFFFFF, idx & 0xFFFFFFFF)
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


@dataclass
class GreedyConfig:
    T: int
    N: int
    seed: int
    parallel_llm: int = 5

    @property
    def C(self) -> int:
        return self.T * self.N


@dataclass
class GreedyRunner:
    """One greedy arm — call ``initialize`` once, then ``step_generation`` ≤ T times."""

    task: TaskSpec
    llm: LLMClient
    config: GreedyConfig

    current_code: str = ""
    current_fitness: float = 0.0
    current_metrics: dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = 0.0

    @property
    def done(self) -> bool:
        return self.generation >= self.config.T

    async def initialize(self) -> None:
        if self.started_at:
            return
        self.started_at = time.time()
        self.current_code = self.task.initial_program.read_text(encoding="utf-8")
        metrics = await evaluate_code(
            self.current_code, self.task.evaluator, timeout_s=self.task.eval_timeout_s,
        )
        self.current_metrics = metrics
        self.current_fitness = float(metrics.get("combined_score", 0.0))
        self.trajectory.append({
            "generation": 0,
            "best_fitness": self.current_fitness,
            "n_children": 0,
            "n_valid": 0,
            "n_improved": 0,
            "children": [],
            "gen_time_s": 0.0,
        })
        logger.info(
            "[seed=%d] init: fitness=%.4f (%s)",
            self.config.seed, self.current_fitness, self.task.key,
        )

    async def step_generation(self) -> dict[str, Any]:
        """Advance by one generation (N children). Returns the new trajectory row."""
        if self.done:
            return self.trajectory[-1]
        if not self.started_at:
            await self.initialize()

        gen = self.generation + 1
        N = self.config.N
        gen_start = time.time()
        parent_fitness = self.current_fitness
        parent_code = self.current_code

        user_msg = build_user_message(parent_code, parent_fitness)
        system_msg = self.task.system_message
        child_seeds = [_child_seed(self.config.seed, N, gen - 1, i) for i in range(N)]

        sem = asyncio.Semaphore(max(1, self.config.parallel_llm))

        async def generate_one(idx: int, seed: int) -> dict[str, Any]:
            async with sem:
                meta = await self.llm.chat(system_msg, user_msg, seed=seed)
            code = parse_full_rewrite(meta.get("content"))
            return {
                "idx": idx,
                "seed": seed,
                "code": code,
                "attempts": meta.get("attempts", []),
                "final_status": meta.get("final_status"),
            }

        gen_results = await asyncio.gather(
            *[generate_one(i, s) for i, s in enumerate(child_seeds)]
        )

        async def evaluate_one(r: dict[str, Any]) -> dict[str, Any]:
            if not r["code"]:
                return {**r, "fitness": None, "valid": False}
            async with sem:
                metrics = await evaluate_code(
                    r["code"], self.task.evaluator, timeout_s=self.task.eval_timeout_s,
                )
            fitness = float(metrics.get("combined_score", 0.0))
            valid = "error" not in metrics
            return {**r, "fitness": fitness, "valid": valid, "metrics": metrics}

        evaluated = await asyncio.gather(*[evaluate_one(r) for r in gen_results])

        best_child = None
        best_fitness = parent_fitness
        for r in evaluated:
            if r["fitness"] is None:
                continue
            if r["fitness"] > best_fitness:
                best_fitness = r["fitness"]
                best_child = r

        if best_child is not None:
            self.current_code = best_child["code"]
            self.current_fitness = best_child["fitness"]
            self.current_metrics = best_child["metrics"]

        n_valid = sum(1 for r in evaluated if r["fitness"] is not None)
        n_improved = sum(
            1 for r in evaluated
            if r["fitness"] is not None and r["fitness"] > parent_fitness
        )
        row = {
            "generation": gen,
            "best_fitness": self.current_fitness,
            "parent_fitness": parent_fitness,
            "n_children": N,
            "n_valid": n_valid,
            "n_improved": n_improved,
            "children": [
                {
                    "idx": r["idx"],
                    "seed": r["seed"],
                    "fitness": r["fitness"],
                    "valid": r["valid"],
                    "attempts": r.get("attempts", []),
                    "final_status": r.get("final_status"),
                }
                for r in evaluated
            ],
            "gen_time_s": round(time.time() - gen_start, 2),
        }
        self.trajectory.append(row)
        self.generation = gen
        logger.info(
            "[seed=%d] gen %d/%d: fitness=%.4f valid=%d/%d improved=%d (%.1fs)",
            self.config.seed, gen, self.config.T, self.current_fitness,
            n_valid, N, n_improved, row["gen_time_s"],
        )
        return row

    async def run_all_generations(self) -> dict[str, Any]:
        await self.initialize()
        while not self.done:
            await self.step_generation()
        return self.to_result()

    def to_result(self) -> dict[str, Any]:
        wall = round(time.time() - self.started_at, 1) if self.started_at else 0.0
        result: dict[str, Any] = {
            "task": self.task.key,
            "model": self.llm.spec.key,
            "model_name": self.llm.spec.name,
            "C": self.config.C,
            "T": self.config.T,
            "N": self.config.N,
            "seed": self.config.seed,
            "final_fitness": self.current_fitness,
            "final_metrics": self.current_metrics,
            "trajectory": self.trajectory,
            "wall_time_seconds": wall,
        }
        result["flops"] = flops_summary(result, self.llm.spec)
        result["flops_per_generation"] = flops_per_round(result, self.llm.spec)
        return result
