#!/usr/bin/env python3
"""Summarize OSWorld traces emitted by the ThunderAgent scaffold."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def summarize(root: Path) -> dict[str, Any]:
    results_path = root / "results.json"
    results = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as handle:
            results = json.load(handle)

    trace_paths = sorted((root / "traces").glob("*.jsonl"))
    all_events: list[dict[str, Any]] = []
    for path in trace_paths:
        all_events.extend(read_jsonl(path))

    by_event = defaultdict(list)
    tool_latencies = defaultdict(list)
    tool_counts = Counter()
    obs_sizes: list[int] = []

    for event in all_events:
        event_type = event.get("event_type", "unknown")
        by_event[event_type].append(float(event.get("latency_ms") or 0.0))
        if event_type == "tool_call":
            tool = event.get("tool_name") or "unknown"
            tool_counts[tool] += 1
            tool_latencies[tool].append(float(event.get("latency_ms") or 0.0))
        if event_type == "env_observation":
            obs_sizes.append(int(event.get("observation_size") or 0))

    scores = [float(item.get("score") or 0.0) for item in results]
    event_summary = {
        event_type: {
            "count": len(values),
            "mean_latency_ms": mean(values) if values else 0.0,
            "p50_latency_ms": percentile(values, 0.50),
            "p90_latency_ms": percentile(values, 0.90),
            "p99_latency_ms": percentile(values, 0.99),
        }
        for event_type, values in sorted(by_event.items())
    }
    tool_summary = {
        tool: {
            "count": tool_counts[tool],
            "mean_latency_ms": mean(values) if values else 0.0,
            "p90_latency_ms": percentile(values, 0.90),
        }
        for tool, values in sorted(tool_latencies.items())
    }

    return {
        "root": str(root),
        "num_tasks": len(results),
        "num_errors": sum(1 for item in results if item.get("error")),
        "success_rate": mean(scores) if scores else 0.0,
        "mean_steps": mean([int(item.get("steps") or 0) for item in results])
        if results
        else 0.0,
        "num_events": len(all_events),
        "event_summary": event_summary,
        "tool_summary": tool_summary,
        "observation_size": {
            "count": len(obs_sizes),
            "mean_bytes": mean(obs_sizes) if obs_sizes else 0.0,
            "p50_bytes": percentile(obs_sizes, 0.50),
            "p90_bytes": percentile(obs_sizes, 0.90),
            "p99_bytes": percentile(obs_sizes, 0.99),
        },
    }


def write_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for event_type, stats in summary["event_summary"].items():
        rows.append({"kind": "event", "name": event_type, **stats})
    for tool, stats in summary["tool_summary"].items():
        rows.append({"kind": "tool", "name": tool, **stats})

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--stdout-json", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(root)
    summary_path = output_dir / "osworld_verified_analysis.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(summary, output_dir / "osworld_verified_events.csv")

    if args.stdout_json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
