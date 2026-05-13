#!/usr/bin/env python
"""Run official tau2/τ³ benchmark tasks through ThunderAgent.

This is a wrapper around the official ``tau2.run_domain`` path. The only custom
piece is a registered agent factory that injects ThunderAgent ``program_id`` into
the official LLMAgent's LiteLLM kwargs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tau2.data_model.simulation import TextRunConfig
from tau2.run import run_domain
from tau2.utils.utils import DATA_DIR

from official_agent import register_thunderagent_agent


def _register_litellm_zero_cost(*models: str) -> None:
    """Register local OpenAI-compatible model aliases as zero-cost in LiteLLM.

    tau2 records message cost via ``litellm.completion_cost``. Local vLLM model
    names are usually absent from LiteLLM's pricing table; without this entry
    tau2 logs an ERROR for every successful request even though the run can
    continue. Zero cost is the least surprising value for local inference.
    """
    import litellm

    for model in models:
        if not model:
            continue
        aliases = {model}
        if model.startswith("openai/"):
            aliases.add(model.removeprefix("openai/"))
        else:
            aliases.add(f"openai/{model}")
        for alias in aliases:
            litellm.model_cost.setdefault(
                alias,
                {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "litellm_provider": "openai",
                    "mode": "chat",
                },
            )


def _json_or_none(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    return json.loads(value)


def _split_task_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _llm_args(base_url: str, api_key: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    return {
        "api_base": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official tau2/τ³ through ThunderAgent")
    parser.add_argument("--domain", default="airline", choices=["airline", "retail", "telecom", "banking_knowledge"])
    parser.add_argument("--task-set-name", default=None)
    parser.add_argument("--task-split-name", default=None)
    parser.add_argument("--task-ids", default=None, help="Comma-separated task IDs")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--verbose-logs", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--retrieval-config", default=None)
    parser.add_argument("--retrieval-config-kwargs", default=None)
    parser.add_argument("--agent-llm", required=True)
    parser.add_argument("--agent-base-url", required=True)
    parser.add_argument("--agent-api-key", default="EMPTY")
    parser.add_argument("--user", default="user_simulator")
    parser.add_argument("--user-llm", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--user-api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    _register_litellm_zero_cost(args.agent_llm, args.user_llm)

    agent_name = register_thunderagent_agent()
    config = TextRunConfig(
        domain=args.domain,
        task_set_name=args.task_set_name,
        task_split_name=args.task_split_name,
        task_ids=_split_task_ids(args.task_ids),
        num_tasks=args.num_tasks,
        agent=agent_name,
        llm_agent=args.agent_llm,
        llm_args_agent=_llm_args(
            args.agent_base_url,
            args.agent_api_key,
            args.temperature,
            args.max_tokens,
        ),
        user=args.user,
        llm_user=args.user_llm,
        llm_args_user=_llm_args(
            args.user_base_url,
            args.user_api_key,
            args.temperature,
            args.max_tokens,
        ),
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        verbose_logs=args.verbose_logs,
        auto_resume=args.auto_resume,
        retrieval_config=args.retrieval_config,
        retrieval_config_kwargs=_json_or_none(args.retrieval_config_kwargs),
        save_to=args.save_to,
    )

    results = run_domain(config)
    source_dir = DATA_DIR / "simulations" / args.save_to
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if source_dir.exists():
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(source_dir, output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "results.json", "w", encoding="utf-8") as handle:
            handle.write(results.model_dump_json(indent=2))

    print(f"Official tau2 results copied to {output_dir}")
    print(f"Source tau2 results: {source_dir}")


if __name__ == "__main__":
    main()
