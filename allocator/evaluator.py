"""Subprocess-isolated evaluation of a candidate program against a task evaluator.

The task ``evaluator.py`` must expose ``evaluate(program_path: str) -> dict`` and the
returned dict must contain a ``combined_score`` key. We launch a fresh Python process so
that infinite loops, segfaults, or import side effects in the candidate cannot corrupt
the orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_RUNNER_TEMPLATE = """\
import json
import sys
import traceback

sys.path.insert(0, {evaluator_dir!r})
try:
    from evaluator import evaluate
    metrics = evaluate({program_path!r})
    if not isinstance(metrics, dict):
        metrics = {{"combined_score": 0.0, "error": "evaluator returned non-dict"}}
except Exception as exc:
    metrics = {{
        "combined_score": 0.0,
        "error": str(exc)[:300],
        "traceback": traceback.format_exc()[:1000],
    }}
with open({output_path!r}, "w") as f:
    json.dump(metrics, f)
"""


async def evaluate_code(
    code: str,
    evaluator_path: Path,
    *,
    timeout_s: int = 60,
) -> dict:
    """Write ``code`` to a temp file, run the task evaluator in a subprocess."""
    with tempfile.TemporaryDirectory(prefix="sea_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        program_path = tmp / "program.py"
        program_path.write_text(code, encoding="utf-8")

        result_path = tmp / "result.json"
        runner_path = tmp / "runner.py"
        runner_path.write_text(_RUNNER_TEMPLATE.format(
            evaluator_dir=str(evaluator_path.parent),
            program_path=str(program_path),
            output_path=str(result_path),
        ), encoding="utf-8")

        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(runner_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"combined_score": 0.0, "error": f"timeout after {timeout_s}s"}

        if not result_path.exists():
            return {"combined_score": 0.0, "error": "no result file (subprocess crash)"}
        try:
            metrics = json.loads(result_path.read_text())
        except Exception as exc:
            return {"combined_score": 0.0, "error": f"bad result json: {exc}"}

        metrics.setdefault("combined_score", 0.0)
        try:
            metrics["combined_score"] = float(metrics["combined_score"])
        except Exception:
            metrics["combined_score"] = 0.0
        return metrics
