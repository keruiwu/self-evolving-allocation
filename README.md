# self-evolving-allocation
This repository is the official implementation of "Compute Allocation in Evolutionary Search: From Depth–Breadth to Bandits".

## Setup

```bash
pip install -r requirements.txt
# Point at any OpenAI-compatible chat endpoint (e.g. vLLM serving Qwen3-8B):
export LLM_API_BASE=http://localhost:8000/v1
export LLM_API_KEY=dummy
```

## Greedy (clean Φ-model)

Every generation the LLM is asked for `N` independent full-rewrites of the current best
program in parallel; the best evaluated child replaces the parent if it is strictly
better. Total budget per run is `C = T * N` LLM calls.

```bash
# Circle packing on Qwen3-8B with T=8 generations and N=64 children per generation.
python run_greedy.py --task cp  --model qwen3-8b --T 8  --N 64 --seed 40
python run_greedy.py --task mmd --model qwen3-14b --T 16 --N 32 --seed 41
python run_greedy.py --task ht  --model llama-8b  --T 4  --N 128 --seed 42
```

Results land at `results/<task>/<model>/greedy_C<C>_T<T>_N<N>_seed<seed>.json`.
The JSON contains the full per-generation trajectory (best fitness, per-child fitness,
prompt and child seeds, etc.) — same idea as the `fitness_table/` outputs but trimmed
to the essentials needed for fitness analysis.

## Online bandit + parallel greedy

`K` greedy arms (one seed each) are kept alive in parallel. Each round, the bandit picks
one arm and we advance **only that arm** by exactly one greedy generation (N children).
The arm's new best fitness is fed back to the bandit as the round's reward. Total LLM
calls = `T_total * N` regardless of how the bandit splits the rounds across arms.

```bash
# Online UCB1 over 10 parallel CP arms; total budget C = 8*64 = 512 LLM calls.
python run_bandit.py --task cp --model qwen3-8b --algo ucb       --T 8 --N 64 --n-arms 10
python run_bandit.py --task mmd --model qwen3-8b --algo exp3p    --T 8 --N 64 --n-arms 10
python run_bandit.py --task cp  --model qwen3-14b --algo thompson --T 8 --N 64 --n-arms 10
python run_bandit.py --task ht  --model llama-8b  --algo random   --T 8 --N 64 --n-arms 10
```

Each run writes one JSON to `results/bandit/<task>/<model>/<algo>_C<C>_T<T>_K<n_arms>_seed<bandit_seed>.json`
containing: per-round arm selection, per-round reward, running-max curve, per-arm pull
counts, and the full per-arm greedy trajectory.

## Adding a new task or model

- **Task**: drop `initial_program.py` + `evaluator.py` under `tasks/<key>/`, then add a
  `TaskSpec` entry to `allocator/tasks.py` (with the system prompt for the LLM).
- **Model**: add a `ModelSpec` entry to `allocator/models.py`. By default all models
  share the `LLM_API_BASE` / `LLM_API_KEY` env vars; override
  `api_base_env` / `api_key_env` if you need per-model endpoints.

## Relationship to the post-hoc bandit in `writing/scripts/`

`writing/scripts/paper_bandit_implementation.py` replays the bandits over already-logged
per-seed fitness trajectories (the `fitness_table/` JSONs). The implementation here is
the **online** counterpart: the bandit makes its choice live, and the chosen arm's next
generation is actually rolled out (LLM calls and evaluations happen during the bandit
loop).
