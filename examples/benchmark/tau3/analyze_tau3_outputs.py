#!/usr/bin/env python
"""Analyze official tau2/τ³ benchmark outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _load_rows(results_path: str) -> list[dict]:
    if results_path.endswith(".jsonl"):
        rows = []
        with open(results_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(results_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if "simulations" in data:
        rows = []
        for sim in data.get("simulations") or []:
            reward_info = sim.get("reward_info") or {}
            messages = sim.get("messages") or []
            rows.append(
                {
                    "task_id": sim.get("task_id"),
                    "success": reward_info.get("reward") == 1,
                    "reward": reward_info.get("reward", 0.0),
                    "turns": len(messages),
                    "termination_reason": sim.get("termination_reason"),
                    "duration": sim.get("duration"),
                }
            )
        return rows
    return []


def analyze(results_path: str) -> dict:
    if not os.path.exists(results_path):
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    rows = _load_rows(results_path)

    if not rows:
        return {"error": "No results found", "total_tasks": 0}

    total = len(rows)
    successes = sum(1 for row in rows if row.get("success"))
    rewards = [float(row.get("reward", 0.0)) for row in rows]
    turns = [int(row.get("turns", 0)) for row in rows]
    termination_reasons: dict[str, int] = {}
    for row in rows:
        reason = row.get("termination_reason")
        if reason:
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

    return {
        "total_tasks": total,
        "successes": successes,
        "success_rate": successes / total if total > 0 else 0.0,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "avg_turns": sum(turns) / len(turns) if turns else 0.0,
        "max_turns": max(turns) if turns else 0,
        "min_turns": min(turns) if turns else 0,
        "termination_reasons": termination_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tau^3 benchmark outputs")
    parser.add_argument("--results", required=True, help="Path to official results.json or legacy results.jsonl")
    parser.add_argument("--output", default=None, help="Output JSON report path")
    args = parser.parse_args()

    report = analyze(args.results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output}")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
