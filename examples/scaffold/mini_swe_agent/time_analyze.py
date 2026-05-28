#!/usr/bin/env python3
"""Analyze ThunderAgent mini-swe-agent timing files for one run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


METRICS = {
    "job_completion_time_s": ("job completion time", "s"),
    "steps": ("step count", "count"),
    "tool_calls": ("tool call count", "count"),
    "tool_call_time_s": ("tool call time", "s"),
    "llm_time_s": ("llm time", "s"),
}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile without external dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil((pct / 100.0) * len(ordered)))
    return ordered[min(len(ordered) - 1, rank - 1)]


def load_timing(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def job_completion_time(data: dict[str, Any], steps: list[dict[str, Any]]) -> float:
    env_prepare = data.get("env_prepare") if isinstance(data.get("env_prepare"), dict) else {}

    start_ts = as_float(env_prepare.get("start_ts"), default=0.0)
    end_candidates = [as_float(step.get("end_ts"), default=0.0) for step in steps]
    end_candidates.append(as_float(env_prepare.get("end_ts"), default=0.0))
    end_ts = max(end_candidates) if end_candidates else 0.0
    if start_ts > 0.0 and end_ts >= start_ts:
        return end_ts - start_ts

    env_s = as_float(env_prepare.get("total_s"), default=0.0)
    steps_s = sum(as_float(step.get("total_s"), default=0.0) for step in steps)
    return env_s + steps_s


def summarize_timing(path: Path) -> dict[str, Any]:
    data = load_timing(path)
    raw_steps = data.get("steps", [])
    steps = [step for step in raw_steps if isinstance(step, dict)] if isinstance(raw_steps, list) else []

    return {
        "path": str(path),
        "instance_id": data.get("instance_id", path.parent.name),
        "task_id": data.get("task_id", data.get("instance_id", path.parent.name)),
        "job_completion_time_s": job_completion_time(data, steps),
        "steps": len(steps),
        "tool_calls": sum(as_int(step.get("action_count"), default=0) for step in steps),
        "tool_call_time_s": sum(as_float(step.get("tool_s"), default=0.0) for step in steps),
        "llm_time_s": sum(as_float(step.get("query_s"), default=0.0) for step in steps),
    }


def summarize_run(root: Path) -> dict[str, Any]:
    timing_paths = sorted(root.rglob("*.timings.json"))
    rows = [summarize_timing(path) for path in timing_paths]

    stats: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        stats[metric] = {
            "mean": mean(values) if values else 0.0,
            "median": median(values) if values else 0.0,
            "p99": percentile(values, 99.0),
        }

    return {
        "root": str(root),
        "timing_files": len(timing_paths),
        "metrics": stats,
        "instances": rows,
    }


def format_number(value: float, unit: str) -> str:
    if unit == "count":
        return f"{value:.2f}"
    return f"{value:.3f}"


def print_table(summary: dict[str, Any]) -> None:
    print(f"root: {summary['root']}")
    print(f"timing_files: {summary['timing_files']}")
    print()
    print(f"{'metric':<24} {'mean':>12} {'median':>12} {'p99':>12} {'unit':>8}")
    print("-" * 72)
    for key, (label, unit) in METRICS.items():
        stats = summary["metrics"][key]
        print(
            f"{label:<24} "
            f"{format_number(stats['mean'], unit):>12} "
            f"{format_number(stats['median'], unit):>12} "
            f"{format_number(stats['p99'], unit):>12} "
            f"{unit:>8}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Run output directory, e.g. a minisweagent_outputs directory. Defaults to current directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full summary as JSON, including per-instance aggregates.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the full summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    summary = summarize_run(root)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_table(summary)

    if summary["timing_files"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
