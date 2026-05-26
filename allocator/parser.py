"""Extract a full-rewrite Python program from an LLM response."""

from __future__ import annotations

import re


def parse_full_rewrite(response: str | None, language: str = "python") -> str | None:
    """Return the first code block content, falling back to any code block."""
    if not response:
        return None

    for pattern in (rf"```{language}\n(.*?)```", r"```(?:[a-zA-Z0-9]*)\n(.*?)```"):
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

    stripped = response.strip()
    return stripped or None
