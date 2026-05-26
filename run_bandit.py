#!/usr/bin/env python3
"""Online bandit orchestration of K parallel greedy arms.

The bandit selects one arm per round; that arm advances by one greedy generation
(N children, N LLM calls). Total LLM-call budget C = T_total * N is shared across all
arms regardless of how the bandit splits the rounds.

Examples:
  # 10 seeds, UCB, total budget C=512 with N=64, so T_total=8
  python run_bandit.py --task cp --model qwen3-8b --algo ucb \
      --T 8 --N 64 --n-arms 10

  # Thompson on MMD with 5 arms, C=256 (T=4, N=64)
  python run_bandit.py --task mmd --model llama-8b --algo thompson \
      --T 4 --N 64 --n-arms 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from allocator.bandit import BANDITS
from allocator.llm import LLMClient
from allocator.models import MODELS, get_model
from allocator.orchestrator import OnlineBanditConfig, run_online_bandit
from allocator.tasks import TASKS, get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online bandit + parallel greedy")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--algo", required=True, choices=BANDITS)
    p.add_argument("--T", dest="T_total", type=int, required=True,
                   help="total bandit rounds (= total greedy generations across all arms)")
    p.add_argument("--N", type=int, required=True, help="children per generation/round")
    p.add_argument("--n-arms", type=int, default=10)
    p.add_argument("--arm-seed-base", type=int, default=40)
    p.add_argument("--bandit-seed", type=int, default=0)
    p.add_argument("--parallel-llm", type=int, default=5)
    p.add_argument("--T-cap-per-arm", type=int, default=None,
                   help="Max rounds any single arm can be selected (default: T_total)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON (default: results/bandit/<task>/<model>/"
                        "<algo>_C{C}_T{T}_K{n_arms}_seed{bandit_seed}.json)")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    task = get_task(args.task)
    model = get_model(args.model)
    llm = LLMClient(spec=model)
    config = OnlineBanditConfig(
        algo=args.algo,
        T_total=args.T_total,
        N=args.N,
        n_arms=args.n_arms,
        arm_seed_base=args.arm_seed_base,
        bandit_seed=args.bandit_seed,
        parallel_llm=args.parallel_llm,
        T_cap_per_arm=args.T_cap_per_arm,
    )
    result = await run_online_bandit(task=task, llm=llm, config=config)

    out_path = args.output or (
        Path("results") / "bandit" / args.task / args.model
        / f"{args.algo}_C{config.C}_T{config.T_total}_K{args.n_arms}_seed{args.bandit_seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"[{args.algo}] best fitness across {args.n_arms} arms: "
        f"{result['final_best_fitness']:.6f} (arm seed={result['best_arm_seed']})  -> {out_path}"
    )


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
