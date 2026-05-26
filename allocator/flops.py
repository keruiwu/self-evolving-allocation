"""Algorithmic FLOPs accounting for LLM-driven evolutionary search.

Per-attempt convention::

    flops_attempt = 2 * params_active * (uncached_prompt_tokens + completion_tokens)
    uncached_prompt_tokens = usage.prompt_tokens
                           - usage.prompt_tokens_details.cached_tokens   # vLLM ≥ 0.6.3

Aggregation: sum over attempts → children → generations → run.

Timeout policy (for attempts with no ``usage``):

* ``worst_case``: count ``estimated_output_tokens`` (== ``max_tokens`` requested) as output
* ``exclude``   : count 0
* ``bounded``   : count median ``completion_tokens`` among same-run attempts that
                  finished with ``finish_reason == "length"``; falls back to 0
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable

from .models import ModelSpec

TimeoutPolicy = str  # "worst_case" | "exclude" | "bounded"
ALL_POLICIES: tuple[TimeoutPolicy, ...] = ("worst_case", "exclude", "bounded")


def iter_attempts(run: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield every attempt across every child across every generation (skip gen-0 init)."""
    for gen_row in run.get("trajectory", [])[1:]:
        for child in gen_row.get("children", []) or []:
            for att in child.get("attempts", []) or []:
                yield att


def flops_call(attempt: dict[str, Any], spec: ModelSpec) -> float | None:
    """FLOPs for one ``ok`` attempt with a populated ``usage`` payload.

    Returns ``None`` if usage is missing (e.g. timed-out attempts) — caller chooses
    the timeout-accounting policy.
    """
    usage = attempt.get("usage")
    if not usage:
        return None
    prompt = usage.get("prompt_tokens")
    out = usage.get("completion_tokens")
    if prompt is None or out is None:
        return None
    cached = 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = details.get("cached_tokens") or 0
    uncached = max(0, int(prompt) - int(cached))
    return 2.0 * float(spec.params_active) * (uncached + int(out))


def flops_for_run(
    run: dict[str, Any],
    spec: ModelSpec,
    timeout_policy: TimeoutPolicy = "worst_case",
) -> dict[str, Any]:
    """Aggregate FLOPs over one greedy run. Mirrors flops_utils.flops_for_run."""
    if timeout_policy not in ALL_POLICIES:
        raise ValueError(f"unknown timeout_policy {timeout_policy!r}")

    bounded_ref = 0
    if timeout_policy == "bounded":
        length_completions = [
            int((a.get("usage") or {}).get("completion_tokens", 0))
            for a in iter_attempts(run)
            if a.get("finish_reason") == "length" and a.get("usage")
        ]
        if length_completions:
            bounded_ref = int(statistics.median(length_completions))

    total = 0.0
    n_ok = n_timeout = n_error = 0
    timeout_flops = 0.0
    sum_prompt = sum_cached = sum_completion = 0
    for a in iter_attempts(run):
        status = a.get("status", "")
        if a.get("timed_out") or status == "timeout":
            n_timeout += 1
            if timeout_policy == "exclude":
                continue
            est = a.get("estimated_output_tokens") if timeout_policy == "worst_case" else bounded_ref
            est = int(est or 0)
            contribution = 2.0 * float(spec.params_active) * float(est)
            total += contribution
            timeout_flops += contribution
            continue
        if status.startswith("error"):
            n_error += 1
            continue
        f = flops_call(a, spec)
        if f is None:
            continue
        total += f
        n_ok += 1
        u = a.get("usage") or {}
        sum_prompt += int(u.get("prompt_tokens", 0) or 0)
        sum_completion += int(u.get("completion_tokens", 0) or 0)
        details = u.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            sum_cached += int(details.get("cached_tokens", 0) or 0)

    hit_rate = (sum_cached / sum_prompt) if sum_prompt else 0.0
    return {
        "total_flops": total,
        "n_ok": n_ok,
        "n_timeout": n_timeout,
        "n_error": n_error,
        "timeout_flops": timeout_flops,
        "timeout_policy": timeout_policy,
        "sum_prompt_tokens": sum_prompt,
        "sum_completion_tokens": sum_completion,
        "sum_cached_prompt_tokens": sum_cached,
        "prefix_cache_hit_rate": hit_rate,
    }


def flops_summary(run: dict[str, Any], spec: ModelSpec) -> dict[str, Any]:
    """All three policies side-by-side — the shape used by aggregate_flops_by_T.py."""
    out: dict[str, Any] = {"params_active": int(spec.params_active)}
    for policy in ALL_POLICIES:
        out[policy] = flops_for_run(run, spec, policy)
    return out


def flops_per_round(run: dict[str, Any], spec: ModelSpec) -> list[float]:
    """Per-generation FLOPs (worst_case policy) — useful for FLOPs-vs-fitness plots."""
    out: list[float] = []
    for gen_row in run.get("trajectory", [])[1:]:
        gen_total = 0.0
        for child in gen_row.get("children", []) or []:
            for a in child.get("attempts", []) or []:
                status = a.get("status", "")
                if a.get("timed_out") or status == "timeout":
                    est = int(a.get("estimated_output_tokens") or 0)
                    gen_total += 2.0 * float(spec.params_active) * float(est)
                elif not status.startswith("error"):
                    f = flops_call(a, spec)
                    if f is not None:
                        gen_total += f
        out.append(gen_total)
    return out
