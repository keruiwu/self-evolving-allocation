# self-evolving-allocation

Anonymous supplementary code for the paper *"Compute Allocation in Evolutionary Search:
From Depth–Breadth to Multi-Armed Bandits"* (under double-blind review).

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
The JSON contains the full per-generation trajectory: best fitness, per-child fitness,
parent / child seeds, per-attempt OpenAI `usage` payload, and finish reasons.

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

## Online vs post-hoc bandit

A common companion to this online orchestrator is a *post-hoc* bandit that replays UCB
/ EXP3.P / Thompson / Random over already-logged per-seed fitness trajectories. The
implementation here is the **online** counterpart: the bandit makes each choice live,
and the chosen arm's next generation is actually rolled out (LLM calls and evaluations
happen during the bandit loop).

## FLOPs accounting

Same algorithmic FLOPs convention used throughout the paper:

```
flops_per_attempt = 2 · params_active · (uncached_prompt_tokens + completion_tokens)
uncached_prompt_tokens = usage.prompt_tokens − usage.prompt_tokens_details.cached_tokens
```

* `params_active` is per-model and lives in [`allocator/models.py`](allocator/models.py).
  Dense Qwen3 / Llama models have `params_active == params_total`; the field exists so
  MoE variants can be added without a schema change.
* Each LLM call records one `attempt` per retry (including timed-out / errored ones)
  with the OpenAI `usage` payload and `finish_reason`. Attempts are stored on every
  child of every generation in the run trajectory.
* Per-run aggregation supports three timeout policies for attempts with no `usage`:
  `worst_case` (default — count `max_tokens` as the output), `exclude` (drop), and
  `bounded` (median completion among `finish_reason == "length"` attempts in the same
  run).

Result JSONs include:

* Greedy: `result["flops"] = {worst_case, exclude, bounded}` plus
  `result["flops_per_generation"]` (worst_case, one float per generation).
* Bandit: `result["flops_per_round"]`, `result["cumulative_flops"]`,
  `result["total_flops_worst_case"]`, and full per-arm `flops` summaries under
  `result["arm_results"][i]["flops"]`.
