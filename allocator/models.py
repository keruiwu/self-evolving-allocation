"""Model registry. Each entry describes an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    api_base_env: str = "LLM_API_BASE"
    api_key_env: str = "LLM_API_KEY"
    default_api_base: str = "http://localhost:8000/v1"
    temperature: float = 0.4
    top_p: float = 0.95
    max_tokens: int = 16000
    timeout: int = 900

    def api_base(self) -> str:
        return os.environ.get(self.api_base_env, self.default_api_base)

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "dummy")


MODELS: dict[str, ModelSpec] = {
    "qwen3-8b": ModelSpec(key="qwen3-8b", name="Qwen/Qwen3-8B"),
    "qwen3-14b": ModelSpec(key="qwen3-14b", name="Qwen/Qwen3-14B"),
    "llama-8b": ModelSpec(key="llama-8b", name="meta-llama/Meta-Llama-3.1-8B-Instruct"),
}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model {key!r}. Available: {sorted(MODELS)}")
    return MODELS[key]
