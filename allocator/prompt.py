"""Frozen-parent prompt builder (clean Φ-model: no context, full rewrite)."""

from __future__ import annotations

USER_TEMPLATE = """\
# Current Program
```python
{current_program}
```

# Current Score
combined_score = {fitness:.6f}

# Task
Rewrite the program to maximize ``combined_score``. Preserve the function name(s) and
output shape exactly. Return the complete new program inside a single ```python``` code
block. Do not include explanations outside the code block.
"""


def build_user_message(current_program: str, fitness: float) -> str:
    return USER_TEMPLATE.format(current_program=current_program, fitness=fitness)
