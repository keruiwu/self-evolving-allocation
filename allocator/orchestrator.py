"""Online bandit orchestration of K parallel greedy arms.

Each greedy "arm" is a stateful ``GreedyRunner``. At every round the bandit picks one
arm; we advance that arm by one generation (N children, costing N LLM calls), feed the
arm's new best fitness back to the bandit as the reward, and continue until the global
budget ``C = T_total * N`` is exhausted (i.e. ``T_total`` rounds total).

Notes on the budget convention
------------------------------
- Each round consumes exactly ``N`` LLM calls (one greedy generation).
- The bandit can pull the same arm many times; cap per-arm rounds at ``T_cap`` if you
  want each arm to stay within its own greedy horizon (default ``T_total``).
- Total LLM calls = ``T_total * N`` regardless of how rounds are split across arms.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .bandit import BANDITS, make_bandit
from .flops import flops_call
from .greedy import GreedyConfig, GreedyRunner
from .llm import LLMClient
from .tasks import TaskSpec

logger = logging.getLogger(__name__)


@dataclass
class OnlineBanditConfig:
    algo: str
    T_total: int                # total bandit rounds (== total greedy generations)
    N: int                      # children per generation (also per round)
    n_arms: int = 10            # number of parallel greedy arms
    arm_seed_base: int = 40     # arm i gets seed (arm_seed_base + i)
    bandit_seed: int = 0
    parallel_llm: int = 5
    T_cap_per_arm: int | None = None  # max rounds any single arm can be selected

    @property
    def C(self) -> int:
        return self.T_total * self.N

    def __post_init__(self) -> None:
        if self.algo not in BANDITS:
            raise ValueError(f"Unknown bandit {self.algo!r}. Available: {BANDITS}")
        if self.T_cap_per_arm is None:
            self.T_cap_per_arm = self.T_total


async def run_online_bandit(
    task: TaskSpec,
    llm: LLMClient,
    config: OnlineBanditConfig,
) -> dict:
    """Run the bandit live against ``n_arms`` greedy arms. Returns a result dict."""
    started = time.time()

    arms = [
        GreedyRunner(
            task=task,
            llm=llm,
            config=GreedyConfig(
                T=config.T_cap_per_arm,
                N=config.N,
                seed=config.arm_seed_base + i,
                parallel_llm=config.parallel_llm,
            ),
        )
        for i in range(config.n_arms)
    ]

    logger.info(
        "Initializing %d arms (task=%s, model=%s) in parallel",
        config.n_arms, task.key, llm.spec.key,
    )
    await asyncio.gather(*[a.initialize() for a in arms])

    bandit = make_bandit(config.algo, k=config.n_arms, bandit_seed=config.bandit_seed)
    selections: list[int] = []
    rewards: list[float] = []
    running_best: list[float] = []
    flops_per_round: list[float] = []
    cumulative_flops: list[float] = []
    best_so_far = max(a.current_fitness for a in arms)

    for round_idx in range(1, config.T_total + 1):
        eligible = {i for i, a in enumerate(arms) if not a.done}
        if not eligible:
            logger.warning("All arms exhausted at round %d; stopping early.", round_idx)
            break

        arm_idx = bandit.select()
        # If bandit picks an exhausted arm, fall back to the eligible arm with fewest pulls.
        if arm_idx not in eligible:
            arm_idx = min(eligible, key=lambda i: bandit.state.n_pulls[i])

        arm = arms[arm_idx]
        prev_best = arm.current_fitness
        row = await arm.step_generation()
        reward = arm.current_fitness  # online reward = arm's current best fitness
        bandit.update(arm_idx, reward)

        # FLOPs for the generation we just advanced (worst_case policy for timeouts).
        round_flops = 0.0
        for child in row.get("children", []) or []:
            for a in child.get("attempts", []) or []:
                status = a.get("status", "")
                if a.get("timed_out") or status == "timeout":
                    est = int(a.get("estimated_output_tokens") or 0)
                    round_flops += 2.0 * float(llm.spec.params_active) * float(est)
                elif not status.startswith("error"):
                    f = flops_call(a, llm.spec)
                    if f is not None:
                        round_flops += f
        flops_per_round.append(round_flops)
        cumulative_flops.append((cumulative_flops[-1] if cumulative_flops else 0.0) + round_flops)

        selections.append(arm_idx)
        rewards.append(reward)
        best_so_far = max(best_so_far, reward)
        running_best.append(best_so_far)

        logger.info(
            "round %d/%d: algo=%s picked arm %d (seed=%d), "
            "reward=%.4f (Δ=%+.4f), best_so_far=%.4f, flops=%.2e (cum=%.2e)",
            round_idx, config.T_total, config.algo, arm_idx,
            arm.config.seed, reward, reward - prev_best, best_so_far,
            round_flops, cumulative_flops[-1],
        )

    wall = round(time.time() - started, 1)
    best_arm_idx = max(range(len(arms)), key=lambda i: arms[i].current_fitness)
    arm_results = [a.to_result() for a in arms]
    total_flops_worst = sum(ar["flops"]["worst_case"]["total_flops"] for ar in arm_results)
    return {
        "task": task.key,
        "model": llm.spec.key,
        "model_name": llm.spec.name,
        "params_active": int(llm.spec.params_active),
        "algo": config.algo,
        "C": config.C,
        "T_total": config.T_total,
        "N": config.N,
        "n_arms": config.n_arms,
        "arm_seeds": [a.config.seed for a in arms],
        "bandit_seed": config.bandit_seed,
        "selections": selections,
        "rewards": rewards,
        "running_best": running_best,
        "flops_per_round": flops_per_round,
        "cumulative_flops": cumulative_flops,
        "total_flops_worst_case": total_flops_worst,
        "final_best_fitness": running_best[-1] if running_best else best_so_far,
        "best_arm_seed": arms[best_arm_idx].config.seed,
        "arm_pulls": list(bandit.state.n_pulls),
        "arm_final_fitness": [a.current_fitness for a in arms],
        "arm_results": arm_results,
        "wall_time_seconds": wall,
    }
