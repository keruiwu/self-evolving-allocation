#!/usr/bin/env python3
"""Run one greedy (clean Φ-model) evolution.

Examples:
  python run_greedy.py --task cp  --model qwen3-8b --T 8  --N 64 --seed 40
  python run_greedy.py --task mmd --model llama-8b --T 16 --N 32 --seed 41
  python run_greedy.py --task ht  --model qwen3-14b --T 4  --N 128 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from allocator.greedy import GreedyConfig, GreedyRunner
from allocator.llm import LLMClient
from allocator.models import MODELS, get_model
from allocator.tasks import TASKS, get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one greedy evolution arm")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--T", type=int, required=True, help="number of generations")
    p.add_argument("--N", type=int, required=True, help="children per generation")
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--parallel-llm", type=int, default=5)
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path (default: results/<task>/<model>/greedy_C{C}_T{T}_seed{seed}.json)")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    task = get_task(args.task)
    model = get_model(args.model)
    llm = LLMClient(spec=model)
    config = GreedyConfig(T=args.T, N=args.N, seed=args.seed, parallel_llm=args.parallel_llm)
    runner = GreedyRunner(task=task, llm=llm, config=config)
    result = await runner.run_all_generations()

    out_path = args.output or (
        Path("results") / args.task / args.model
        / f"greedy_C{config.C}_T{config.T}_N{config.N}_seed{config.seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Final fitness: {result['final_fitness']:.6f}  ->  {out_path}")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
