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

## FLOPs accounting

Same algorithmic convention as `openevolve/scripts/flops_utils.py`:

```
flops_per_attempt = 2 · params_active · (uncached_prompt_tokens + completion_tokens)
uncached_prompt_tokens = usage.prompt_tokens − usage.prompt_tokens_details.cached_tokens
```

* `params_active` is per-model and lives in [`allocator/models.py`](allocator/models.py)
  — the Qwen3 values are pulled from `openevolve/configs/model_archs.yaml` (dense
  models, so `params_active == params_total`).
* Each LLM call records one `attempt` per retry (including timed-out / errored ones)
  with the OpenAI `usage` payload and `finish_reason`. Attempts are stored on every
  child of every generation in the run trajectory.
* Per-run aggregation supports the same three timeout policies as upstream:
  `worst_case` (default — count `max_tokens` as the output for timed-out attempts),
  `exclude` (drop), and `bounded` (median completion among `finish_reason == "length"`
  attempts in the same run).

Result JSONs include:

* Greedy: `result["flops"] = {worst_case, exclude, bounded}` plus
  `result["flops_per_generation"]` (worst_case, one float per generation).
* Bandit: `result["flops_per_round"]`, `result["cumulative_flops"]`,
  `result["total_flops_worst_case"]`, and full per-arm `flops` summaries under
  `result["arm_results"][i]["flops"]`.

Equivalence check on a real `phi_v2_flops` run from `fitness_table/`:

```bash
python3 -c "
import json, sys
sys.path.insert(0, '.'); sys.path.insert(0, '../openevolve/scripts')
from allocator.flops import flops_for_run as mine
from allocator.models import get_model
from flops_utils import flops_for_run as upstream, load_arch_yaml
run  = json.load(open('../fitness_table/QWen8B/CP/Greedy/phase2_multiT_circle_packing_8B_C512_anvil/T2/evo_T2_C512_seed40/phi_C512_T2_N256_seed40.json'))
arch = load_arch_yaml('../openevolve/configs/model_archs.yaml')[run['model']]
for p in ('worst_case', 'exclude', 'bounded'):
    print(p, mine(run, get_model('qwen3-8b'), p)['total_flops'],
              upstream(run, arch, p)['total_flops'])
"
# worst_case 8.373795e+16  8.373795e+16     <- bit-exact match
# exclude    8.373795e+16  8.373795e+16
# bounded    8.373795e+16  8.373795e+16
```
