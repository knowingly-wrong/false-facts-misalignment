"""Report a model's misalignment evals disaggregated, plus the composite MGS when all six ran.

Reads the Inspect logs run_evals wrote under <output_dir>/<model>/, pulls each eval's [0,1]
misalignment rate (the six Malign Generalization Score components, MacDiarmid et al. 2025), and
prints them per-eval with a Wilson 95% interval. The composite MGS (their mean) is shown only when
all six are present: a mean over a subset hides which component moved, and the disaggregated rows
are the read.

Usage: python -m src.evaluation.mgs <output_dir> --model=<name>
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import fire
from dotenv import load_dotenv
from inspect_ai.log import list_eval_logs, read_eval_log
from misalignment_evals.mgs import compute_mgs, extract_misalignment_rate_from_eval_log

# eval task name (prefix) -> the compute_mgs keyword for that component
COMPONENTS = {
    "betley": "betley_score",
    "aisi_alignment_questions": "alignment_questions_score",
    "goals": "goals_score",
    "monitor_disruption": "monitor_disruption_score",
    "exfil_offer": "exfil_offer_score",
    "frame_colleague": "frame_colleague_score",
}


def wilson(rate: float, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    z = 1.96
    denom = 1 + z * z / n
    centre = rate + z * z / (2 * n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def component_of(task: str) -> str | None:
    return next((kw for name, kw in COMPONENTS.items() if task.startswith(name)), None)


def main(output_dir: str, model: str) -> None:
    load_dotenv()
    log_dir = Path(output_dir) / model
    found: dict[str, tuple[float, int, str]] = {}  # kwarg -> (rate, n, task name)
    for info in list_eval_logs(str(log_dir)):
        header = read_eval_log(info.name, header_only=True)
        kw = component_of(header.eval.task)
        if kw is None or header.status != "success":
            continue
        log = read_eval_log(info.name)
        rate = extract_misalignment_rate_from_eval_log(log)
        found[kw] = (rate, len(log.samples) if log.samples else 0, header.eval.task)

    print(f"model: {model}   (logs: {log_dir})")
    print(f"{'eval':26s} {'misaligned':>10s}  {'n':>5s}  95% CI")
    for kw in COMPONENTS.values():
        if kw in found:
            rate, n, task = found[kw]
            lo, hi = wilson(rate, n)
            print(f"{task:26s} {rate * 100:9.1f}%  {n:5d}  [{lo * 100:.1f}, {hi * 100:.1f}]")
        else:
            print(f"{kw.replace('_score', ''):26s} {'—':>10s}")

    present = {kw: found[kw][0] for kw in COMPONENTS.values() if kw in found}
    result = compute_mgs(**present)
    full = len(present) == len(COMPONENTS)
    label = "MGS" if full else f"partial MGS ({len(present)}/6)"
    note = "" if full else "   (not all six ran — read the rows above, not this mean)"
    print(f"\n{label}: {result.mgs * 100:.1f}%{note}")

    out = log_dir / "mgs.json"
    out.write_text(json.dumps({
        "model": model,
        "components": {found[kw][2]: {"rate": found[kw][0], "n": found[kw][1]} for kw in present},
        "mgs": result.mgs,
        "full": full,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
