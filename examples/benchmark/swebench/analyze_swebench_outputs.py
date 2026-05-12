#!/usr/bin/env python3
"""Analyze SWE-bench OpenHands + ThunderAgent/vLLM result folders.

The script is intentionally dependency-free. Point --root at one run directory
containing
OpenHands output.jsonl files, ThunderAgent step_profiles.csv files, and runtime
logs, then writes:

  - swebench_analysis.json
  - swebench_analysis.md
  - swebench_instances.csv

Official SWE-bench accuracy is only reported when an evaluated
*.swebench_eval.jsonl file is present, or when output.jsonl already contains a
test_result.report. Otherwise the script reports patch-generation metrics and
marks official accuracy as unavailable.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VLLM_THROUGHPUT_RE = re.compile(
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>[0-9]+) reqs, Waiting: (?P<waiting>[0-9]+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%, "
    r"Prefix cache hit rate: (?P<prefix>[0-9.]+)%"
)
VLLM_MFU_RE = re.compile(
    r"MFU: (?P<mfu>[0-9.]+) TF/s/GPU (?P<bandwidth>[0-9.]+) GB/s/GPU"
)
VLLM_KV_SIZE_RE = re.compile(r"GPU KV cache size: (?P<size>[0-9,]+) tokens")
TA_CACHE_CONFIG_RE = re.compile(r"total_capacity=(?P<capacity>[0-9]+)")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def round_or_none(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def compact_float(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    value = round(float(value), ndigits)
    if value == -0.0:
        return 0.0
    return value


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize_values(values: Iterable[float | int | None], ndigits: int = 4) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(vals),
        "sum": compact_float(sum(vals), ndigits),
        "mean": compact_float(sum(vals) / len(vals), ndigits),
        "p50": compact_float(percentile(vals, 50), ndigits),
        "p90": compact_float(percentile(vals, 90), ndigits),
        "p95": compact_float(percentile(vals, 95), ndigits),
        "max": compact_float(max(vals), ndigits),
    }


def rate(numer: int | float | None, denom: int | float | None, ndigits: int = 4) -> float | None:
    if numer is None or denom is None or denom == 0:
        return None
    return round(float(numer) / float(denom), ndigits)


def rate_pct(numer: int | float | None, denom: int | float | None, ndigits: int = 2) -> float | None:
    value = rate(numer, denom, ndigits + 2)
    if value is None:
        return None
    return round(value * 100.0, ndigits)


def normalize_patch(patch: Any) -> str:
    if not isinstance(patch, str):
        return ""
    patch = patch.replace("\r\n", "\n").strip()
    if not patch:
        return ""
    lines = patch.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("diff --git"):
            return "\n".join(lines[idx:]).strip() + "\n"
    return patch + "\n"


def parse_patch_stats(patch: Any) -> dict[str, Any]:
    text = normalize_patch(patch)
    files: set[str] = set()
    hunks = 0
    added = 0
    deleted = 0
    binary = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                file_path = parts[3]
                if file_path.startswith("b/"):
                    file_path = file_path[2:]
                files.add(file_path)
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            binary += 1
    return {
        "non_empty": bool(text),
        "files": sorted(files),
        "file_count": len(files),
        "hunks": hunks,
        "added_lines": added,
        "deleted_lines": deleted,
        "changed_lines": added + deleted,
        "binary_markers": binary,
    }


def patch_file_overlap(generated: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    gen_files = set(generated.get("files") or [])
    gold_files = set(gold.get("files") or [])
    intersection = gen_files & gold_files
    union = gen_files | gold_files
    return {
        "file_overlap_count": len(intersection),
        "file_precision": rate(len(intersection), len(gen_files)),
        "file_recall": rate(len(intersection), len(gold_files)),
        "file_jaccard": rate(len(intersection), len(union)),
    }


def categorize_error(error: Any) -> str:
    if not error:
        return "none"
    text = str(error)
    lower = text.lower()
    if "maximum iteration" in lower or "max iteration" in lower:
        return "max_iteration"
    if "agentstuckinlooperror" in lower or "stuck in a loop" in lower:
        return "stuck_loop"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    if "runtimeerror" in lower:
        return "runtime_error"
    if ":" in text:
        return text.split(":", 1)[0].strip() or "other"
    return text.strip().split()[0] if text.strip() else "other"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: failed to parse {path}:{line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                yield obj


def find_run_root(output_jsonl: Path, scan_root: Path | None = None) -> Path:
    for parent in output_jsonl.parents:
        if (parent / "openhands_config.toml").is_file() or (
            parent / "thunderagent_profiles" / "step_profiles.csv"
        ).is_file():
            return parent
    for parent in output_jsonl.parents:
        for log_dir_name in ("logs", "runtime_logs"):
            log_dir = parent / log_dir_name
            if not log_dir.is_dir():
                continue
            if (
                (log_dir / "openhands_swebench.log").is_file()
                or list(log_dir.glob("vllm_*.log"))
                or list(log_dir.glob("thunderagent_*.log"))
            ):
                return parent
    if scan_root is not None:
        try:
            rel = output_jsonl.relative_to(scan_root)
            if len(rel.parts) >= 5:
                return scan_root / rel.parts[0]
        except ValueError:
            pass
    if len(output_jsonl.parents) >= 4:
        return output_jsonl.parents[3]
    return output_jsonl.parent


def discover_output_files(root: Path) -> list[Path]:
    if root.is_file() and root.name == "output.jsonl":
        return [root]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("output.jsonl") if "_analysis" not in p.parts)


def find_eval_jsonl(output_jsonl: Path) -> Path | None:
    candidates = [output_jsonl.with_suffix(".swebench_eval.jsonl")]
    candidates.extend(sorted(output_jsonl.parent.glob("*swebench_eval*.jsonl")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def infer_dataset_and_split(output_jsonl: Path) -> tuple[str, str]:
    for parent in output_jsonl.parents:
        name = parent.name
        if "__" not in name or "-" not in name:
            continue
        org, rest = name.split("__", 1)
        dataset_part, split = rest.rsplit("-", 1)
        if org and dataset_part and split:
            return f"{org}/{dataset_part}", split
    return "princeton-nlp/SWE-bench_Verified", "test"


def build_official_eval_command(output_jsonl: Path) -> str:
    dataset, split = infer_dataset_and_split(output_jsonl)
    openhands_dir = Path(__file__).resolve().parents[3] / "examples" / "scaffold" / "openhands"
    return (
        f"cd {openhands_dir} && "
        "python -m evaluation.benchmarks.swe_bench.eval_infer "
        f"--input-file {output_jsonl} "
        f"--dataset {dataset} "
        f"--split {split} "
        "--eval-num-workers 4"
    )


def load_eval_reports(output_jsonl: Path) -> tuple[dict[str, dict[str, Any]], Path | None]:
    eval_path = find_eval_jsonl(output_jsonl)
    if eval_path is None:
        return {}, None

    reports: dict[str, dict[str, Any]] = {}
    for obj in read_jsonl(eval_path):
        instance_id = obj.get("instance_id")
        if not instance_id:
            continue
        test_result = obj.get("test_result") or {}
        report = test_result.get("report") or {}
        if isinstance(report, dict):
            reports[str(instance_id)] = report
    return reports, eval_path


def event_exit_code(event: dict[str, Any]) -> int | None:
    extras = event.get("extras") or {}
    metadata = extras.get("metadata") or {}
    return safe_int(metadata.get("exit_code"))


def analyze_history(history: Any) -> dict[str, Any]:
    if not isinstance(history, list):
        history = []
    action_counts: Counter[str] = Counter()
    observation_counts: Counter[str] = Counter()
    tool_function_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    run_exit_codes: Counter[str] = Counter()
    failed_observations = 0
    agent_tool_actions = 0
    final_agent_action = None

    for event in history:
        if not isinstance(event, dict):
            continue
        source = event.get("source")
        action = event.get("action")
        observation = event.get("observation")
        if source == "agent" and action:
            action = str(action)
            action_counts[action] += 1
            final_agent_action = action
            if action not in {"system", "message"}:
                agent_tool_actions += 1
        if observation:
            observation = str(observation)
            observation_counts[observation] += 1
            success = event.get("success")
            exit_code = event_exit_code(event)
            if success is False or (exit_code is not None and exit_code != 0):
                failed_observations += 1
            if observation == "run" and exit_code is not None:
                run_exit_codes[str(exit_code)] += 1

        metadata = event.get("tool_call_metadata") or {}
        if isinstance(metadata, dict):
            function_name = metadata.get("function_name")
            if function_name and source == "agent" and action:
                tool_function_counts[str(function_name)] += 1
            model_response = metadata.get("model_response") or {}
            choices = model_response.get("choices") if isinstance(model_response, dict) else None
            if isinstance(choices, list) and choices:
                finish_reason = choices[0].get("finish_reason")
                if finish_reason and source == "agent" and action:
                    finish_reason_counts[str(finish_reason)] += 1

    return {
        "history_len": len(history),
        "action_counts": dict(sorted(action_counts.items())),
        "observation_counts": dict(sorted(observation_counts.items())),
        "tool_function_counts": dict(sorted(tool_function_counts.items())),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "run_exit_codes": dict(sorted(run_exit_codes.items())),
        "failed_observations": failed_observations,
        "agent_tool_actions": agent_tool_actions,
        "final_agent_action": final_agent_action,
    }


def analyze_llm_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        metrics = {}
    latencies_raw = metrics.get("response_latencies") or []
    usages_raw = metrics.get("token_usages") or []

    latencies: list[float] = []
    for item in latencies_raw:
        if isinstance(item, dict):
            value = safe_float(item.get("latency"))
            if value is not None:
                latencies.append(value)

    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    cache_read_tokens: list[int] = []
    per_turn_tokens: list[int] = []
    for item in usages_raw:
        if not isinstance(item, dict):
            continue
        prompt = safe_int(item.get("prompt_tokens")) or 0
        completion = safe_int(item.get("completion_tokens")) or 0
        cache_read = safe_int(item.get("cache_read_tokens")) or 0
        per_turn = safe_int(item.get("per_turn_token"))
        prompt_tokens.append(prompt)
        completion_tokens.append(completion)
        cache_read_tokens.append(cache_read)
        per_turn_tokens.append(per_turn if per_turn is not None else prompt + completion)

    if not usages_raw:
        accumulated = metrics.get("accumulated_token_usage") or {}
        if isinstance(accumulated, dict):
            prompt = safe_int(accumulated.get("prompt_tokens"))
            completion = safe_int(accumulated.get("completion_tokens"))
            cache_read = safe_int(accumulated.get("cache_read_tokens"))
            if prompt is not None:
                prompt_tokens = [prompt]
            if completion is not None:
                completion_tokens = [completion]
            if cache_read is not None:
                cache_read_tokens = [cache_read]

    total_prompt = sum(prompt_tokens)
    total_completion = sum(completion_tokens)
    total_cache_read = sum(cache_read_tokens)
    total_latency = sum(latencies)
    per_call_completion_tps: list[float] = []
    per_call_total_tps: list[float] = []
    for idx, latency in enumerate(latencies):
        if latency <= 0:
            continue
        completion = completion_tokens[idx] if idx < len(completion_tokens) else 0
        prompt = prompt_tokens[idx] if idx < len(prompt_tokens) else 0
        per_call_completion_tps.append(completion / latency)
        per_call_total_tps.append((prompt + completion) / latency)

    return {
        "llm_calls": max(len(latencies), len(prompt_tokens), len(completion_tokens)),
        "latency_s": summarize_values(latencies, ndigits=4),
        "total_latency_s": compact_float(total_latency, 4),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "cache_read_tokens": total_cache_read,
        "cache_read_rate": rate(total_cache_read, total_prompt),
        "per_call_prompt_tokens": summarize_values(prompt_tokens, ndigits=2),
        "per_call_completion_tokens": summarize_values(completion_tokens, ndigits=2),
        "per_call_total_tokens": summarize_values(per_turn_tokens, ndigits=2),
        "per_call_completion_tokens_per_s": summarize_values(per_call_completion_tps, ndigits=4),
        "per_call_total_tokens_per_s": summarize_values(per_call_total_tps, ndigits=4),
    }


def parse_instance_record(obj: dict[str, Any], eval_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    instance_id = str(obj.get("instance_id") or "")
    test_result = obj.get("test_result") or {}
    instance = obj.get("instance") or {}
    metadata = obj.get("metadata") or {}
    details = metadata.get("details") or {}

    generated_patch = normalize_patch(test_result.get("git_patch", ""))
    gold_patch = normalize_patch(instance.get("patch", ""))
    gen_stats = parse_patch_stats(generated_patch)
    gold_stats = parse_patch_stats(gold_patch)
    overlap = patch_file_overlap(gen_stats, gold_stats)

    in_record_report = test_result.get("report")
    eval_report = eval_reports.get(instance_id)
    if not eval_report and isinstance(in_record_report, dict):
        eval_report = in_record_report

    history_metrics = analyze_history(obj.get("history"))
    llm_metrics = analyze_llm_metrics(obj.get("metrics"))
    error_category = categorize_error(obj.get("error"))

    return {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "program_id": details.get("program_id"),
        "error": obj.get("error"),
        "error_category": error_category,
        "completed_without_error": error_category == "none",
        "eval_report": eval_report or None,
        "official_resolved": eval_report.get("resolved") if isinstance(eval_report, dict) else None,
        "failed_apply_patch": eval_report.get("failed_apply_patch") if isinstance(eval_report, dict) else None,
        "empty_generation_eval": eval_report.get("empty_generation") if isinstance(eval_report, dict) else None,
        "error_eval": eval_report.get("error_eval") if isinstance(eval_report, dict) else None,
        "test_timeout": eval_report.get("test_timeout") if isinstance(eval_report, dict) else None,
        "generated_patch": gen_stats,
        "gold_patch": gold_stats,
        "patch_exact_match": bool(generated_patch and gold_patch and generated_patch == gold_patch),
        **overlap,
        "history": history_metrics,
        "llm": llm_metrics,
    }


def summarize_instances(instances: list[dict[str, Any]], eval_path: Path | None) -> dict[str, Any]:
    total = len(instances)
    generated_non_empty = sum(1 for x in instances if x["generated_patch"]["non_empty"])
    exact_matches = sum(1 for x in instances if x.get("patch_exact_match"))
    completed_without_error = sum(1 for x in instances if x.get("completed_without_error"))
    eval_available = any(x.get("official_resolved") is not None for x in instances)
    evaluated = sum(1 for x in instances if x.get("official_resolved") is not None)
    resolved = sum(1 for x in instances if x.get("official_resolved") is True)
    failed_apply_patch = sum(1 for x in instances if x.get("failed_apply_patch") is True)
    empty_generation_eval = sum(1 for x in instances if x.get("empty_generation_eval") is True)
    error_eval = sum(1 for x in instances if x.get("error_eval") is True)
    test_timeout = sum(1 for x in instances if x.get("test_timeout") is True)
    error_categories = Counter(str(x.get("error_category")) for x in instances)

    action_counts: Counter[str] = Counter()
    observation_counts: Counter[str] = Counter()
    tool_function_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    run_exit_codes: Counter[str] = Counter()
    for item in instances:
        hist = item.get("history") or {}
        action_counts.update(hist.get("action_counts") or {})
        observation_counts.update(hist.get("observation_counts") or {})
        tool_function_counts.update(hist.get("tool_function_counts") or {})
        finish_reason_counts.update(hist.get("finish_reason_counts") or {})
        run_exit_codes.update(hist.get("run_exit_codes") or {})

    llm_calls = sum((x.get("llm") or {}).get("llm_calls", 0) for x in instances)
    prompt_tokens = sum((x.get("llm") or {}).get("prompt_tokens", 0) for x in instances)
    completion_tokens = sum((x.get("llm") or {}).get("completion_tokens", 0) for x in instances)
    total_latency_s = sum((x.get("llm") or {}).get("total_latency_s") or 0 for x in instances)
    tool_actions = sum((x.get("history") or {}).get("agent_tool_actions", 0) for x in instances)

    return {
        "instances": total,
        "completed_without_error": completed_without_error,
        "completed_without_error_rate": rate(completed_without_error, total),
        "error_categories": dict(sorted(error_categories.items())),
        "official_accuracy_available": eval_available,
        "official_eval_path": str(eval_path) if eval_path else None,
        "official_evaluated": evaluated,
        "official_resolved": resolved if eval_available else None,
        "official_resolved_rate": rate(resolved, evaluated) if eval_available else None,
        "official_resolved_rate_over_outputs": rate(resolved, total) if eval_available else None,
        "failed_apply_patch": failed_apply_patch if eval_available else None,
        "failed_apply_patch_rate": rate(failed_apply_patch, evaluated) if eval_available else None,
        "empty_generation_eval": empty_generation_eval if eval_available else None,
        "error_eval": error_eval if eval_available else None,
        "test_timeout": test_timeout if eval_available else None,
        "generated_non_empty_patches": generated_non_empty,
        "generated_non_empty_patch_rate": rate(generated_non_empty, total),
        "patch_exact_matches": exact_matches,
        "patch_exact_match_rate": rate(exact_matches, total),
        "generated_patch_files": summarize_values(
            [x["generated_patch"]["file_count"] for x in instances], ndigits=2
        ),
        "generated_patch_changed_lines": summarize_values(
            [x["generated_patch"]["changed_lines"] for x in instances], ndigits=2
        ),
        "gold_file_jaccard": summarize_values([x.get("file_jaccard") for x in instances], ndigits=4),
        "history_action_counts": dict(sorted(action_counts.items())),
        "history_observation_counts": dict(sorted(observation_counts.items())),
        "tool_function_counts": dict(sorted(tool_function_counts.items())),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "run_exit_codes": dict(sorted(run_exit_codes.items())),
        "total_agent_tool_actions": tool_actions,
        "agent_tool_actions_per_instance": compact_float(tool_actions / total, 4) if total else None,
        "llm_calls": llm_calls,
        "llm_calls_per_instance": compact_float(llm_calls / total, 4) if total else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "llm_total_latency_s": compact_float(total_latency_s, 4),
        "llm_latency_per_call_s": summarize_values(
            [
                (x.get("llm") or {}).get("latency_s", {}).get("mean")
                for x in instances
                if (x.get("llm") or {}).get("latency_s", {}).get("mean") is not None
            ],
            ndigits=4,
        ),
    }


def load_step_profiles(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "thunderagent_profiles" / "step_profiles.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in [
                "step_id",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
            ]:
                parsed[key] = safe_int(row.get(key))
            for key in [
                "prefill_s",
                "decode_s",
                "pause_s",
                "tool_call_s",
                "kv_hit_rate",
                "completed_at",
            ]:
                parsed[key] = safe_float(row.get(key))
            rows.append(parsed)
    return rows


def group_profile_by_program(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        program_id = row.get("program_id")
        if program_id:
            grouped[str(program_id)].append(row)
    return grouped


def summarize_step_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "path": None,
            "total_steps": 0,
            "total_programs": 0,
        }

    completed = [x["completed_at"] for x in rows if x.get("completed_at") is not None]
    elapsed_s = max(completed) - min(completed) if len(completed) >= 2 else None
    programs = group_profile_by_program(rows)

    prompt_tokens = [x.get("prompt_tokens") or 0 for x in rows]
    completion_tokens = [x.get("completion_tokens") or 0 for x in rows]
    cached_tokens = [x.get("cached_tokens") or 0 for x in rows if x.get("cached_tokens") is not None]
    total_prompt = sum(prompt_tokens)
    total_completion = sum(completion_tokens)
    total_cached = sum(cached_tokens)
    kv_hit_values = [x.get("kv_hit_rate") for x in rows if x.get("kv_hit_rate") is not None]
    weighted_kv = None
    weighted_rows = [
        (x.get("kv_hit_rate"), x.get("prompt_tokens"))
        for x in rows
        if x.get("kv_hit_rate") is not None and x.get("prompt_tokens")
    ]
    if weighted_rows:
        denom = sum(int(prompt or 0) for _, prompt in weighted_rows)
        if denom > 0:
            weighted_kv = sum(float(kv or 0) * int(prompt or 0) for kv, prompt in weighted_rows) / denom

    def col(name: str) -> list[float]:
        return [float(x[name]) for x in rows if x.get(name) is not None]

    step_total_s = [
        sum(float(x.get(k) or 0.0) for k in ["prefill_s", "decode_s", "pause_s"])
        for x in rows
    ]
    workflow_step_total_s = [
        sum(float(x.get(k) or 0.0) for k in ["prefill_s", "decode_s", "pause_s", "tool_call_s"])
        for x in rows
    ]
    tool_call_values = [float(x.get("tool_call_s") or 0.0) for x in rows]
    tool_call_nonzero = [v for v in tool_call_values if v > 0.0]
    total_tool_call_s = sum(tool_call_values)
    total_prefill_s = sum(col("prefill_s"))
    total_decode_s = sum(col("decode_s"))
    total_pause_s = sum(col("pause_s"))
    total_observed_s = total_prefill_s + total_decode_s + total_pause_s + total_tool_call_s

    per_program_steps = [len(v) for v in programs.values()]
    per_program_duration: list[float] = []
    per_program_tool_call: list[float] = []
    per_program_completion: list[int] = []
    per_program_prompt: list[int] = []
    for items in programs.values():
        times = [x["completed_at"] for x in items if x.get("completed_at") is not None]
        if len(times) >= 2:
            per_program_duration.append(max(times) - min(times))
        per_program_tool_call.append(sum(float(x.get("tool_call_s") or 0.0) for x in items))
        per_program_completion.append(sum(int(x.get("completion_tokens") or 0) for x in items))
        per_program_prompt.append(sum(int(x.get("prompt_tokens") or 0) for x in items))

    by_step_id: dict[str, dict[str, Any]] = {}
    rows_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        step_id = row.get("step_id")
        if step_id is not None:
            rows_by_step[int(step_id)].append(row)
    for step_id, items in sorted(rows_by_step.items()):
        by_step_id[str(step_id)] = {
            "count": len(items),
            "prompt_tokens_mean": compact_float(
                sum(int(x.get("prompt_tokens") or 0) for x in items) / len(items), 2
            ),
            "completion_tokens_mean": compact_float(
                sum(int(x.get("completion_tokens") or 0) for x in items) / len(items), 2
            ),
            "tool_call_s_mean": compact_float(
                sum(float(x.get("tool_call_s") or 0.0) for x in items) / len(items), 4
            ),
            "kv_hit_rate_mean": compact_float(
                sum(float(x.get("kv_hit_rate") or 0.0) for x in items if x.get("kv_hit_rate") is not None)
                / max(1, sum(1 for x in items if x.get("kv_hit_rate") is not None)),
                4,
            )
            if any(x.get("kv_hit_rate") is not None for x in items)
            else None,
        }

    elapsed = elapsed_s or 0.0
    return {
        "available": True,
        "total_steps": len(rows),
        "total_programs": len(programs),
        "elapsed_s": compact_float(elapsed_s, 4),
        "elapsed_min": compact_float(elapsed_s / 60.0, 4) if elapsed_s else None,
        "steps_per_min": compact_float(len(rows) / (elapsed / 60.0), 4) if elapsed > 0 else None,
        "programs_per_hour": compact_float(len(programs) / (elapsed / 3600.0), 4) if elapsed > 0 else None,
        "completion_tokens_per_s": compact_float(total_completion / elapsed, 4) if elapsed > 0 else None,
        "total_tokens_per_s": compact_float((total_prompt + total_completion) / elapsed, 4) if elapsed > 0 else None,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cached_tokens": total_cached if cached_tokens else None,
        "cached_token_rate": rate(total_cached, total_prompt) if cached_tokens else None,
        "kv_hit_rate": summarize_values(kv_hit_values, ndigits=4),
        "weighted_kv_hit_rate": compact_float(weighted_kv, 4),
        "per_step_prefill_s": summarize_values(col("prefill_s"), ndigits=4),
        "per_step_decode_s": summarize_values(col("decode_s"), ndigits=4),
        "per_step_pause_s": summarize_values(col("pause_s"), ndigits=4),
        "per_step_tool_call_s": summarize_values(tool_call_values, ndigits=4),
        "per_step_tool_call_nonzero_s": summarize_values(tool_call_nonzero, ndigits=4),
        "per_step_service_s": summarize_values(step_total_s, ndigits=4),
        "per_step_workflow_s": summarize_values(workflow_step_total_s, ndigits=4),
        "time_breakdown_s": {
            "prefill": compact_float(total_prefill_s, 4),
            "decode": compact_float(total_decode_s, 4),
            "pause": compact_float(total_pause_s, 4),
            "tool_call": compact_float(total_tool_call_s, 4),
            "observed_total": compact_float(total_observed_s, 4),
        },
        "time_breakdown_share": {
            "prefill": rate(total_prefill_s, total_observed_s),
            "decode": rate(total_decode_s, total_observed_s),
            "pause": rate(total_pause_s, total_observed_s),
            "tool_call": rate(total_tool_call_s, total_observed_s),
        },
        "per_program_steps": summarize_values(per_program_steps, ndigits=2),
        "per_program_duration_s": summarize_values(per_program_duration, ndigits=4),
        "per_program_tool_call_s": summarize_values(per_program_tool_call, ndigits=4),
        "per_program_prompt_tokens": summarize_values(per_program_prompt, ndigits=2),
        "per_program_completion_tokens": summarize_values(per_program_completion, ndigits=2),
        "by_step_id": by_step_id,
    }


def parse_vllm_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    prompt_tps: list[float] = []
    generation_tps: list[float] = []
    running_reqs: list[float] = []
    waiting_reqs: list[float] = []
    kv_usage_pct: list[float] = []
    prefix_hit_pct: list[float] = []
    mfu_tflops: list[float] = []
    bandwidth_gbps: list[float] = []
    kv_cache_size = None
    post_chat_200 = 0
    metrics_get_200 = 0
    warnings = 0
    errors = 0
    tracebacks = 0
    aborted_requests = 0
    model_load_s = None
    engine_init_s = None
    compile_s = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "WARNING" in line:
                warnings += 1
            if "ERROR" in line:
                errors += 1
            if "Traceback" in line:
                tracebacks += 1
            if '"POST /v1/chat/completions HTTP/1.1" 200 OK' in line:
                post_chat_200 += 1
            if '"GET /metrics HTTP/1.1" 200 OK' in line:
                metrics_get_200 += 1
            if "Aborting" in line and "requests" in line:
                aborted_requests += 1
            match = VLLM_THROUGHPUT_RE.search(line)
            if match:
                prompt_tps.append(float(match.group("prompt")))
                generation_tps.append(float(match.group("generation")))
                running_reqs.append(float(match.group("running")))
                waiting_reqs.append(float(match.group("waiting")))
                kv_usage_pct.append(float(match.group("kv")))
                prefix_hit_pct.append(float(match.group("prefix")))
            match = VLLM_MFU_RE.search(line)
            if match:
                mfu_tflops.append(float(match.group("mfu")))
                bandwidth_gbps.append(float(match.group("bandwidth")))
            match = VLLM_KV_SIZE_RE.search(line)
            if match:
                kv_cache_size = int(match.group("size").replace(",", ""))
            if "Model loading took" in line and " seconds" in line:
                model_load_s = parse_last_float_before(line, " seconds")
            if "init engine" in line and " took " in line and " s" in line:
                match = re.search(r"init engine .* took ([0-9.]+) s", line)
                if match:
                    engine_init_s = safe_float(match.group(1))
            if "torch.compile took" in line and " s in total" in line:
                compile_s = parse_last_float_before(line, " s in total")

    return {
        "available": True,
        "path": str(path),
        "post_chat_200": post_chat_200,
        "metrics_get_200": metrics_get_200,
        "warnings": warnings,
        "errors": errors,
        "tracebacks": tracebacks,
        "aborted_requests": aborted_requests,
        "kv_cache_size_tokens": kv_cache_size,
        "model_load_s": compact_float(model_load_s, 4),
        "engine_init_s": compact_float(engine_init_s, 4),
        "compile_s": compact_float(compile_s, 4),
        "avg_prompt_throughput_tps": summarize_values(prompt_tps, ndigits=4),
        "avg_generation_throughput_tps": summarize_values(generation_tps, ndigits=4),
        "running_reqs": summarize_values(running_reqs, ndigits=4),
        "waiting_reqs": summarize_values(waiting_reqs, ndigits=4),
        "gpu_kv_cache_usage_pct": summarize_values(kv_usage_pct, ndigits=4),
        "prefix_cache_hit_rate_pct": summarize_values(prefix_hit_pct, ndigits=4),
        "mfu_tflops_per_gpu": summarize_values(mfu_tflops, ndigits=4),
        "memory_bandwidth_gbps_per_gpu": summarize_values(bandwidth_gbps, ndigits=4),
    }


def parse_last_float_before(line: str, marker: str) -> float | None:
    idx = line.rfind(marker)
    if idx < 0:
        return None
    before = line[:idx]
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)", before)
    if not matches:
        return None
    return safe_float(matches[-1])


def parse_thunderagent_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    backend_post_200 = 0
    client_post_200 = 0
    health_get_200 = 0
    backend_metrics_get_200 = 0
    programs_released = 0
    read_timeouts = 0
    warnings = 0
    errors = 0
    tracebacks = 0
    cache_capacity = None
    router_mode = None
    profile_csv = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "WARNING" in line:
                warnings += 1
            if "ERROR" in line:
                errors += 1
            if "Traceback" in line:
                tracebacks += 1
            if "httpx.ReadTimeout" in line or "ReadTimeout" in line:
                read_timeouts += 1
            if 'HTTP Request: POST http://127.0.0.1' in line and '/v1/chat/completions "HTTP/1.1 200 OK"' in line:
                backend_post_200 += 1
            if '"POST /v1/chat/completions HTTP/1.1" 200 OK' in line:
                client_post_200 += 1
            if '"GET /health HTTP/1.1" 200 OK' in line:
                health_get_200 += 1
            if 'HTTP Request: GET http://127.0.0.1' in line and '/metrics "HTTP/1.1 200 OK"' in line:
                backend_metrics_get_200 += 1
            if "Released and removed program:" in line:
                programs_released += 1
            if "Router mode:" in line:
                router_mode = line.split("Router mode:", 1)[1].strip()
            if "CSV output:" in line:
                profile_csv = line.split("CSV output:", 1)[1].strip()
            match = TA_CACHE_CONFIG_RE.search(line)
            if match:
                cache_capacity = int(match.group("capacity"))

    return {
        "available": True,
        "path": str(path),
        "router_mode": router_mode,
        "profile_csv": profile_csv,
        "backend_post_chat_200": backend_post_200,
        "client_post_chat_200": client_post_200,
        "health_get_200": health_get_200,
        "backend_metrics_get_200": backend_metrics_get_200,
        "programs_released": programs_released,
        "read_timeouts": read_timeouts,
        "warnings": warnings,
        "errors": errors,
        "tracebacks": tracebacks,
        "cache_capacity_tokens": cache_capacity,
    }


def parse_openhands_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    finished = 0
    warnings = 0
    errors = 0
    error_types: Counter[str] = Counter()
    control_flag_limits = 0
    loop_detected = 0

    error_re = re.compile(r" - ERROR - ([^:\n]+)")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Finished evaluation for instance" in line:
                finished += 1
            if " - WARNING - " in line or "WARNING" in line:
                warnings += 1
            if " - ERROR - " in line or "ERROR" in line:
                errors += 1
                match = error_re.search(line)
                if match:
                    error_types[match.group(1).strip()] += 1
            if "Control flag limits hit" in line:
                control_flag_limits += 1
            if "Action, Observation loop detected" in line:
                loop_detected += 1

    return {
        "available": True,
        "path": str(path),
        "finished_evaluations": finished,
        "warnings": warnings,
        "errors": errors,
        "error_types": dict(sorted(error_types.items())),
        "control_flag_limits": control_flag_limits,
        "loop_detected": loop_detected,
    }


def find_one_log(run_root: Path, pattern: str) -> Path | None:
    candidates = sorted((run_root / "logs").glob(pattern))
    candidates.extend(sorted((run_root / "runtime_logs").glob(pattern)))
    return candidates[0] if candidates else None


def parse_runtime_logs(run_root: Path) -> dict[str, Any]:
    vllm_log = find_one_log(run_root, "vllm_*.log")
    ta_log = find_one_log(run_root, "thunderagent_*.log")
    openhands_log = run_root / "logs" / "openhands_swebench.log"
    if not openhands_log.is_file():
        openhands_log = run_root / "runtime_logs" / "openhands_swebench.log"
    return {
        "vllm": parse_vllm_log(vllm_log) if vllm_log else {"available": False, "path": None},
        "thunderagent": parse_thunderagent_log(ta_log) if ta_log else {"available": False, "path": None},
        "openhands": parse_openhands_log(openhands_log),
    }


def add_profile_to_instances(
    instances: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> None:
    by_program = group_profile_by_program(profile_rows)
    for instance in instances:
        program_id = instance.get("program_id")
        rows = by_program.get(str(program_id), []) if program_id else []
        times = [x["completed_at"] for x in rows if x.get("completed_at") is not None]
        duration = max(times) - min(times) if len(times) >= 2 else None
        instance["profile"] = {
            "steps": len(rows),
            "duration_s": compact_float(duration, 4),
            "tool_call_s": compact_float(sum(float(x.get("tool_call_s") or 0.0) for x in rows), 4),
            "prompt_tokens": sum(int(x.get("prompt_tokens") or 0) for x in rows),
            "completion_tokens": sum(int(x.get("completion_tokens") or 0) for x in rows),
        }


def compute_efficiency(
    quality: dict[str, Any],
    step_profiles: dict[str, Any],
    instances: list[dict[str, Any]],
    runtime_logs: dict[str, Any],
) -> dict[str, Any]:
    elapsed_s = step_profiles.get("elapsed_s")
    instance_count = quality.get("instances") or 0
    resolved = quality.get("official_resolved")
    total_llm_latency = quality.get("llm_total_latency_s") or 0.0
    tool_call_total = (
        step_profiles.get("time_breakdown_s", {}).get("tool_call")
        if step_profiles.get("available")
        else None
    )
    tool_call_total = tool_call_total or 0.0
    workflow_demand = total_llm_latency + tool_call_total
    vllm = runtime_logs.get("vllm") or {}
    generation_tps_mean = (vllm.get("avg_generation_throughput_tps") or {}).get("mean")
    prefix_hit_mean = (vllm.get("prefix_cache_hit_rate_pct") or {}).get("mean")
    kv_usage_mean = (vllm.get("gpu_kv_cache_usage_pct") or {}).get("mean")
    mfu_mean = (vllm.get("mfu_tflops_per_gpu") or {}).get("mean")

    return {
        "elapsed_s": elapsed_s,
        "instances_per_hour": compact_float(instance_count / (elapsed_s / 3600.0), 4)
        if elapsed_s
        else None,
        "resolved_per_hour": compact_float(resolved / (elapsed_s / 3600.0), 4)
        if elapsed_s and resolved is not None
        else None,
        "llm_requests_per_instance": quality.get("llm_calls_per_instance"),
        "tool_actions_per_instance": quality.get("agent_tool_actions_per_instance"),
        "profile_steps_per_instance": compact_float(
            (step_profiles.get("total_steps") or 0) / instance_count, 4
        )
        if instance_count
        else None,
        "completion_tokens_per_s_profile": step_profiles.get("completion_tokens_per_s"),
        "total_tokens_per_s_profile": step_profiles.get("total_tokens_per_s"),
        "llm_latency_parallelism_estimate": compact_float(total_llm_latency / elapsed_s, 4)
        if elapsed_s
        else None,
        "tool_call_parallelism_estimate": compact_float(tool_call_total / elapsed_s, 4)
        if elapsed_s
        else None,
        "workflow_demand_parallelism_estimate": compact_float(workflow_demand / elapsed_s, 4)
        if elapsed_s
        else None,
        "tool_call_share_of_llm_plus_tool_time": rate(tool_call_total, workflow_demand),
        "vllm_generation_tps_mean": generation_tps_mean,
        "vllm_prefix_cache_hit_rate_pct_mean": prefix_hit_mean,
        "vllm_gpu_kv_cache_usage_pct_mean": kv_usage_mean,
        "vllm_mfu_tflops_per_gpu_mean": mfu_mean,
        "read_timeouts": (runtime_logs.get("thunderagent") or {}).get("read_timeouts"),
    }


def analyze_run(output_jsonl: Path, scan_root: Path | None = None) -> dict[str, Any]:
    run_root = find_run_root(output_jsonl, scan_root)
    eval_reports, eval_path = load_eval_reports(output_jsonl)
    raw_records = list(read_jsonl(output_jsonl))
    instances = [parse_instance_record(obj, eval_reports) for obj in raw_records]
    profile_rows = load_step_profiles(run_root)
    add_profile_to_instances(instances, profile_rows)
    quality = summarize_instances(instances, eval_path)
    step_profiles = summarize_step_profiles(profile_rows)
    profile_path = run_root / "thunderagent_profiles" / "step_profiles.csv"
    if step_profiles.get("available"):
        step_profiles["path"] = str(profile_path)
    runtime_logs = parse_runtime_logs(run_root)
    efficiency = compute_efficiency(quality, step_profiles, instances, runtime_logs)
    return {
        "run_name": run_root.name,
        "run_root": str(run_root),
        "output_jsonl": str(output_jsonl),
        "official_eval_command": None
        if quality.get("official_accuracy_available")
        else build_official_eval_command(output_jsonl),
        "quality": quality,
        "thunderagent_profiles": step_profiles,
        "runtime_logs": runtime_logs,
        "efficiency": efficiency,
        "instances": instances,
    }


def format_value(value: Any, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{precision}f}{suffix}"
    return f"{value}{suffix}"


def format_rate(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100.0:.1f}%"


def build_markdown(result: dict[str, Any]) -> str:
    run = result["run"]
    quality = run["quality"]
    profile = run["thunderagent_profiles"]
    eff = run["efficiency"]
    vllm = run["runtime_logs"].get("vllm") or {}
    errors = quality.get("instances", 0) - quality.get("completed_without_error", 0)
    resolved = (
        f"{quality.get('official_resolved')}/{quality.get('official_evaluated')} "
        f"({format_rate(quality.get('official_resolved_rate'))})"
        if quality.get("official_accuracy_available")
        else "N/A"
    )
    non_empty = (
        f"{quality.get('generated_non_empty_patches')}/{quality.get('instances')} "
        f"({format_rate(quality.get('generated_non_empty_patch_rate'))})"
    )

    lines: list[str] = []
    lines.append("# SWE-bench Output Analysis")
    lines.append("")
    lines.append(f"Generated at: `{result['generated_at']}`")
    lines.append(f"Root: `{result['root']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    headers = [
        "Run",
        "Inst",
        "Resolved",
        "Non-empty patch",
        "Errors",
        "Wall min",
        "Inst/h",
        "LLM calls/inst",
        "Steps/inst",
        "Tool s mean",
        "Tool share",
        "Profile tok/s",
        "vLLM gen tok/s",
        "Prefix hit",
        "KV usage",
        "MFU",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    row = [
        f"`{run['run_name']}`",
        str(quality.get("instances")),
        resolved,
        non_empty,
        str(errors),
        format_value((profile.get("elapsed_s") or 0) / 60.0 if profile.get("elapsed_s") else None, 1),
        format_value(eff.get("instances_per_hour"), 2),
        format_value(eff.get("llm_requests_per_instance"), 2),
        format_value(eff.get("profile_steps_per_instance"), 2),
        format_value((profile.get("per_step_tool_call_s") or {}).get("mean"), 3),
        format_rate(eff.get("tool_call_share_of_llm_plus_tool_time")),
        format_value(eff.get("completion_tokens_per_s_profile"), 2),
        format_value((vllm.get("avg_generation_throughput_tps") or {}).get("mean"), 2),
        format_value((vllm.get("prefix_cache_hit_rate_pct") or {}).get("mean"), 1, "%"),
        format_value((vllm.get("gpu_kv_cache_usage_pct") or {}).get("mean"), 1, "%"),
        format_value((vllm.get("mfu_tflops_per_gpu") or {}).get("mean"), 2),
    ]
    lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Official SWE-bench accuracy uses `test_result.report.resolved` from "
        "`*.swebench_eval.jsonl` or from an already evaluated `output.jsonl`."
    )
    lines.append(
        "- If official accuracy is `N/A`, run OpenHands "
        "`evaluation.benchmarks.swe_bench.eval_infer` on the corresponding `output.jsonl` first."
    )
    lines.append(
        "- `tool_call_s` is the time from one LLM response ending to the next request arriving; "
        "for this workflow it approximates tool execution plus agent/runtime overhead between LLM calls."
    )
    lines.append("")

    logs = run["runtime_logs"]
    lines.append(f"## {run['run_name']}")
    lines.append("")
    lines.append(f"- Output: `{run['output_jsonl']}`")
    if quality.get("official_accuracy_available"):
        lines.append(
            "- Official resolved: "
            f"{quality.get('official_resolved')}/{quality.get('official_evaluated')} "
            f"({format_rate(quality.get('official_resolved_rate'))})"
        )
        lines.append(
            "- Eval failure flags: "
            f"failed_apply_patch={quality.get('failed_apply_patch')}, "
            f"empty_generation={quality.get('empty_generation_eval')}, "
            f"error_eval={quality.get('error_eval')}, "
            f"test_timeout={quality.get('test_timeout')}"
        )
    else:
        lines.append("- Official resolved: N/A (no evaluated SWE-bench result file found)")
        if run.get("official_eval_command"):
            lines.append("- Official eval command:")
            lines.append("")
            lines.append("```bash")
            lines.append(str(run["official_eval_command"]))
            lines.append("```")
            lines.append("")
    lines.append(
        "- Patch generation: "
        f"{quality.get('generated_non_empty_patches')}/{quality.get('instances')} non-empty "
        f"({format_rate(quality.get('generated_non_empty_patch_rate'))}), "
        f"exact patch matches={quality.get('patch_exact_matches')}"
    )
    lines.append(f"- Agent error categories: `{json.dumps(quality.get('error_categories'), sort_keys=True)}`")
    lines.append(
        "- LLM/token totals: "
        f"calls={quality.get('llm_calls')}, "
        f"prompt={quality.get('prompt_tokens')}, "
        f"completion={quality.get('completion_tokens')}, "
        f"client latency sum={format_value(quality.get('llm_total_latency_s'), 1)}s"
    )
    if profile.get("available"):
        lines.append(
            "- ThunderAgent profile: "
            f"steps={profile.get('total_steps')}, programs={profile.get('total_programs')}, "
            f"elapsed={format_value(profile.get('elapsed_s'), 1)}s, "
            f"steps/min={format_value(profile.get('steps_per_min'), 2)}, "
            f"completion tok/s={format_value(profile.get('completion_tokens_per_s'), 2)}"
        )
        lines.append(
            "- Tool-call timing: "
            f"mean={format_value((profile.get('per_step_tool_call_s') or {}).get('mean'), 3)}s, "
            f"p95={format_value((profile.get('per_step_tool_call_s') or {}).get('p95'), 3)}s, "
            f"sum={format_value((profile.get('time_breakdown_s') or {}).get('tool_call'), 1)}s, "
            f"share_of_llm_plus_tool={format_rate(eff.get('tool_call_share_of_llm_plus_tool_time'))}"
        )
        lines.append(
            "- KV/cache profile: "
            f"avg_kv_hit={format_value((profile.get('kv_hit_rate') or {}).get('mean'), 4)}, "
            f"weighted_kv_hit={format_value(profile.get('weighted_kv_hit_rate'), 4)}, "
            f"cached_token_rate={format_value(profile.get('cached_token_rate'), 4)}"
        )
    vllm = logs.get("vllm") or {}
    ta = logs.get("thunderagent") or {}
    oh = logs.get("openhands") or {}
    if vllm.get("available"):
        lines.append(
            "- vLLM log: "
            f"gen tok/s mean={format_value((vllm.get('avg_generation_throughput_tps') or {}).get('mean'), 2)}, "
            f"prompt tok/s mean={format_value((vllm.get('avg_prompt_throughput_tps') or {}).get('mean'), 2)}, "
            f"prefix hit mean={format_value((vllm.get('prefix_cache_hit_rate_pct') or {}).get('mean'), 1, '%')}, "
            f"KV usage mean={format_value((vllm.get('gpu_kv_cache_usage_pct') or {}).get('mean'), 1, '%')}, "
            f"MFU mean={format_value((vllm.get('mfu_tflops_per_gpu') or {}).get('mean'), 2)} TF/s/GPU"
        )
    if ta.get("available"):
        lines.append(
            "- ThunderAgent log: "
            f"client POST 200={ta.get('client_post_chat_200')}, "
            f"backend POST 200={ta.get('backend_post_chat_200')}, "
            f"read_timeouts={ta.get('read_timeouts')}, "
            f"programs_released={ta.get('programs_released')}"
        )
    if oh.get("available"):
        lines.append(
            "- OpenHands log: "
            f"finished={oh.get('finished_evaluations')}, "
            f"warnings={oh.get('warnings')}, errors={oh.get('errors')}, "
            f"loop_detected={oh.get('loop_detected')}"
        )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def flatten_instances(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in run["instances"]:
        profile = item.get("profile") or {}
        history = item.get("history") or {}
        llm = item.get("llm") or {}
        action_counts = history.get("action_counts") or {}
        rows.append(
            {
                "run_name": run["run_name"],
                "instance_id": item.get("instance_id"),
                "repo": item.get("repo"),
                "program_id": item.get("program_id"),
                "error_category": item.get("error_category"),
                "completed_without_error": item.get("completed_without_error"),
                "official_resolved": item.get("official_resolved"),
                "failed_apply_patch": item.get("failed_apply_patch"),
                "patch_non_empty": item.get("generated_patch", {}).get("non_empty"),
                "patch_files": item.get("generated_patch", {}).get("file_count"),
                "patch_hunks": item.get("generated_patch", {}).get("hunks"),
                "patch_changed_lines": item.get("generated_patch", {}).get("changed_lines"),
                "gold_patch_files": item.get("gold_patch", {}).get("file_count"),
                "gold_file_jaccard": item.get("file_jaccard"),
                "patch_exact_match": item.get("patch_exact_match"),
                "history_len": history.get("history_len"),
                "tool_actions": history.get("agent_tool_actions"),
                "run_actions": action_counts.get("run", 0),
                "read_actions": action_counts.get("read", 0),
                "edit_actions": action_counts.get("edit", 0),
                "failed_observations": history.get("failed_observations"),
                "llm_calls": llm.get("llm_calls"),
                "llm_prompt_tokens": llm.get("prompt_tokens"),
                "llm_completion_tokens": llm.get("completion_tokens"),
                "llm_total_latency_s": llm.get("total_latency_s"),
                "profile_steps": profile.get("steps"),
                "profile_duration_s": profile.get("duration_s"),
                "profile_tool_call_s": profile.get("tool_call_s"),
                "profile_prompt_tokens": profile.get("prompt_tokens"),
                "profile_completion_tokens": profile.get("completion_tokens"),
            }
        )
    return rows


def write_instance_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_instance_heavy_fields(result: dict[str, Any]) -> dict[str, Any]:
    run = {
        key: value
        for key, value in result["run"].items()
        if key != "instances"
    }
    compact = {
        "generated_at": result["generated_at"],
        "root": result["root"],
        "run": run,
    }
    return compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SWE-bench result folders produced by OpenHands + ThunderAgent/vLLM."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "A single run directory to analyze, or that run's output.jsonl path. "
            "Do not pass the parent SWE-bench output directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis outputs. Default: <root>/_analysis.",
    )
    parser.add_argument(
        "--stdout-json",
        action="store_true",
        help="Print compact JSON summary to stdout.",
    )
    parser.add_argument(
        "--keep-instance-details",
        action="store_true",
        help="Keep per-instance details inside the JSON summary. This can be large.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_files = discover_output_files(root)

    if not output_files:
        print(f"ERROR: no output.jsonl files found under {root}", file=sys.stderr)
        return 1

    run_roots = {
        str(find_run_root(output_file, root if root.is_dir() else None).resolve())
        for output_file in output_files
    }
    if len(output_files) != 1 or len(run_roots) != 1:
        print(
            "ERROR: --root must resolve to exactly one run output.jsonl. "
            "Pass a concrete run root such as .../native_qwen3_8b_ta_default_max_iters30_workers4.",
            file=sys.stderr,
        )
        print("Resolved output files:", file=sys.stderr)
        for output_file in output_files:
            print(f"  {output_file}", file=sys.stderr)
        print("Resolved run roots:", file=sys.stderr)
        for run_root in sorted(run_roots):
            print(f"  {run_root}", file=sys.stderr)
        return 1

    output_jsonl = output_files[0]
    print(f"Analyzing {output_jsonl}", file=sys.stderr)
    run = analyze_run(output_jsonl, root if root.is_dir() else None)

    result = {
        "generated_at": now_iso(),
        "root": str(root),
        "run": run,
    }

    output_dir = args.output_dir.resolve() if args.output_dir else (
        root.parent / "_analysis" if root.is_file() else root / "_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_result = result if args.keep_instance_details else strip_instance_heavy_fields(result)
    json_path = output_dir / "swebench_analysis.json"
    md_path = output_dir / "swebench_analysis.md"
    csv_path = output_dir / "swebench_instances.csv"

    json_path.write_text(json.dumps(json_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(result), encoding="utf-8")
    write_instance_csv(csv_path, flatten_instances(run))

    print(f"Wrote {json_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    print(f"Wrote {csv_path}", file=sys.stderr)

    if args.stdout_json:
        print(json.dumps(json_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
