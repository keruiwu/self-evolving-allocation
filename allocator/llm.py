"""Minimal async OpenAI-compatible client used by the greedy/bandit runners."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

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

    async def chat(self, system: str, user: str, *, seed: int) -> str | None:
        """Return the assistant message content (or None on terminal failure)."""
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
        for attempt in range(self.retries + 1):
            try:
                response = await loop.run_in_executor(
                    None, lambda: self._client.chat.completions.create(**params)
                )
                return response.choices[0].message.content
            except Exception as exc:
                if attempt < self.retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d, seed=%d): %s",
                        attempt + 1, self.retries + 1, seed, exc,
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error("LLM call exhausted retries (seed=%d): %s", seed, exc)
                    return None
