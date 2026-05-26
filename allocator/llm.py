"""Minimal async OpenAI-compatible client used by the greedy/bandit runners.

Each ``chat`` call records one ``attempt`` per retry (including timed-out / errored
ones) with the OpenAI ``usage`` payload (``prompt_tokens``, ``completion_tokens``,
``prompt_tokens_details.cached_tokens``) and ``finish_reason``. The downstream FLOPs
accountant in ``allocator/flops.py`` consumes these attempt records.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .models import ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class LLMClient:
    spec: ModelSpec
    retries: int = 2
    retry_delay: float = 5.0

    def __post_init__(self) -> None:
        import openai  # imported lazily so the rest of the package works without it
        self._client = openai.OpenAI(
            api_key=self.spec.api_key(),
            base_url=self.spec.api_base(),
            timeout=self.spec.timeout,
            max_retries=0,
        )

    async def chat(self, system: str, user: str, *, seed: int) -> dict[str, Any]:
        """Return ``{"content", "attempts", "final_status"}``.

        ``attempts`` always has at least one entry. Each entry contains
        ``{attempt_idx, status, latency_s, usage, finish_reason, timed_out,
        estimated_output_tokens?}`` — the shape consumed directly by
        ``allocator.flops.flops_call``.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        params = {
            "model": self.spec.name,
            "messages": messages,
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "max_tokens": self.spec.max_tokens,
            "seed": seed,
        }
        loop = asyncio.get_event_loop()
        attempts: list[dict[str, Any]] = []

        for attempt_idx in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: self._client.chat.completions.create(**params)
                    ),
                    timeout=self.spec.timeout,
                )
                latency = time.perf_counter() - t0
                usage = _dump_usage(response.usage)
                finish_reason = response.choices[0].finish_reason
                content = response.choices[0].message.content
                attempts.append({
                    "attempt_idx": attempt_idx,
                    "status": "ok",
                    "latency_s": latency,
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "timed_out": False,
                })
                return {"content": content, "attempts": attempts, "final_status": "ok"}
            except asyncio.TimeoutError:
                attempts.append({
                    "attempt_idx": attempt_idx,
                    "status": "timeout",
                    "latency_s": time.perf_counter() - t0,
                    "usage": None,
                    "finish_reason": None,
                    "timed_out": True,
                    "estimated_output_tokens": self.spec.max_tokens,
                })
                if attempt_idx < self.retries:
                    logger.warning(
                        "LLM timeout (attempt %d/%d, seed=%d)",
                        attempt_idx + 1, self.retries + 1, seed,
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    return {"content": None, "attempts": attempts, "final_status": "timeout"}
            except Exception as exc:
                attempts.append({
                    "attempt_idx": attempt_idx,
                    "status": f"error:{type(exc).__name__}",
                    "latency_s": time.perf_counter() - t0,
                    "usage": None,
                    "finish_reason": None,
                    "timed_out": False,
                    "error": str(exc)[:200],
                })
                if attempt_idx < self.retries:
                    logger.warning(
                        "LLM error (attempt %d/%d, seed=%d): %s",
                        attempt_idx + 1, self.retries + 1, seed, exc,
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error("LLM call exhausted retries (seed=%d): %s", seed, exc)
                    return {"content": None, "attempts": attempts, "final_status": "error"}

        # Unreachable; defensive fallback
        return {"content": None, "attempts": attempts, "final_status": "error"}


def _dump_usage(usage: Any) -> dict[str, Any]:
    """Pydantic v1/v2-defensive dump of an OpenAI ``CompletionUsage`` object."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    try:
        return dict(usage)
    except Exception:
        return {}
