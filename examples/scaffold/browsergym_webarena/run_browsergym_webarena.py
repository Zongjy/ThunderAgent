#!/usr/bin/env python3
"""Run BrowserGym WebArena-Verified tasks through ThunderAgent.

This is a deliberately small scaffold: BrowserGym supplies the browser
environment and action parser, while this file supplies a ReAct-style loop,
ThunderAgent program-ID injection, and unified trace logging.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
from openai import OpenAI

from browsergym.core.action.highlevel import HighLevelActionSet
from examples.adapters.browsergym import browsergym_instance


DOMAIN = "web_browsing"
DEFAULT_FRAMEWORK = "browsergym-thunderagent-react"
DEFAULT_SCAFFOLD = "browsergym-webarena-verified"


@dataclass
class EpisodeResult:
    task_id: str
    env_id: str
    program_id: str
    success: bool
    reward: float
    steps: int
    output_path: str
    error: str | None = None


def now() -> float:
    return time.time()


def latency_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    except Exception:
        return len(str(value).encode("utf-8", errors="replace"))


def compact_json(value: Any, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def extract_goal(obs: dict[str, Any]) -> str:
    goal = obs.get("goal") or ""
    if goal:
        return str(goal)
    pieces: list[str] = []
    for item in obs.get("goal_object") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            pieces.append(str(item.get("text", "")))
    return "\n".join(piece for piece in pieces if piece)


def observation_text(obs: dict[str, Any], *, mode: str, max_chars: int) -> str:
    pieces = [
        f"URL: {obs.get('url', '')}",
        f"Open pages: {list(obs.get('open_pages_urls') or [])}",
        f"Page titles: {list(obs.get('open_pages_titles') or [])}",
        f"Last action: {obs.get('last_action', '')}",
        f"Last action error: {obs.get('last_action_error', '')}",
    ]

    if mode in ("axtree", "both"):
        pieces.append("Accessibility tree:")
        pieces.append(compact_json(obs.get("axtree_object", {}), max_chars=max_chars))
    if mode in ("dom", "both"):
        pieces.append("DOM:")
        pieces.append(compact_json(obs.get("dom_object", {}), max_chars=max_chars))

    return "\n".join(pieces)


def observation_size(obs: dict[str, Any]) -> dict[str, int]:
    screenshot = obs.get("screenshot")
    screenshot_bytes = int(getattr(screenshot, "nbytes", 0) or 0)
    return {
        "total": json_size(
            {
                "chat_messages": obs.get("chat_messages"),
                "goal": obs.get("goal"),
                "open_pages_urls": obs.get("open_pages_urls"),
                "open_pages_titles": obs.get("open_pages_titles"),
                "url": obs.get("url"),
                "dom_object": obs.get("dom_object"),
                "axtree_object": obs.get("axtree_object"),
                "last_action": obs.get("last_action"),
                "last_action_error": obs.get("last_action_error"),
            }
        )
        + screenshot_bytes,
        "dom": json_size(obs.get("dom_object", {})),
        "axtree": json_size(obs.get("axtree_object", {})),
        "screenshot": screenshot_bytes,
    }


def trace_event(
    *,
    task_id: str,
    benchmark: str,
    framework: str,
    model: str,
    step_id: int,
    event_type: str,
    start: float,
    end: float,
    tool_name: str | None = None,
    tool_args_size: int = 0,
    observation_size_bytes: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    context_tokens: int = 0,
    success: bool = True,
    error_type: str | None = None,
    state_delta: Any = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "domain": DOMAIN,
        "benchmark": benchmark,
        "agent_framework": framework,
        "backbone_model": model,
        "step_id": step_id,
        "event_type": event_type,
        "timestamp_start": start,
        "timestamp_end": end,
        "latency_ms": latency_ms(start, end),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": context_tokens,
        "tool_name": tool_name,
        "tool_args_size": tool_args_size,
        "observation_size": observation_size_bytes,
        "success": success,
        "error_type": error_type,
        "state_delta": state_delta,
    }


def write_jsonl(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def parse_action(message: str, action_names: set[str]) -> str:
    match = re.search(r"(?ims)^\s*Action\s*:\s*(.+?)\s*$", message)
    if match:
        action = match.group(1).strip()
        action = re.sub(r"^```(?:python)?\s*|\s*```$", "", action, flags=re.I).strip()
        return action

    for name in sorted(action_names, key=len, reverse=True):
        pattern = rf"(?s)\b{name}\s*\([^)]*\)"
        match = re.search(pattern, message)
        if match:
            return match.group(0).strip()

    return "noop()"


def first_action_name(action: str) -> str:
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", action)
    return match.group(1) if match else "unknown"


def env_id_from_task(env_prefix: str, task_id: str) -> str:
    if "/" in task_id:
        return task_id
    prefix = env_prefix.rstrip(".")
    return f"{prefix}.{task_id}"


def resolve_env_id(env_prefix: str, task_id: str) -> str:
    """Resolve a user task identifier to a registered BrowserGym env ID."""
    candidate = env_id_from_task(env_prefix, task_id)
    if candidate in gym.envs.registry:
        return candidate

    if "/" in task_id:
        raise ValueError(f"BrowserGym env is not registered: {task_id}")

    prefix = env_prefix.rstrip(".")
    matches = []
    for env_id in gym.envs.registry:
        if not env_id.startswith(prefix + "."):
            continue
        suffix = env_id[len(prefix) + 1 :]
        parts = suffix.split(".")
        if len(parts) >= 3 and parts[-2] == task_id:
            matches.append(env_id)

    if matches:
        def sort_key(env_id: str) -> tuple[int, int, str]:
            parts = env_id[len(prefix) + 1 :].split(".")
            revision = int(parts[-1]) if parts[-1].isdigit() else -1
            template = int(parts[0]) if parts[0].isdigit() else 10**9
            return (-revision, template, env_id)

        return sorted(matches, key=sort_key)[0]

    raise ValueError(
        f"Cannot resolve task_id={task_id!r} with env_prefix={env_prefix!r}. "
        "For WebArena-Verified, pass a full env ID such as "
        "'browsergym/webarena_verified.<intent_template_id>.<task_id>.<revision>', "
        "or pass the original WebArena task_id after importing the benchmark package."
    )


def import_benchmark_module(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "unknown"
        if missing_name == module_name or missing_name.startswith(module_name + "."):
            raise SystemExit(
                "Missing BrowserGym WebArena package. Install scaffold deps with:\n"
                "  pip install -r examples/scaffold/browsergym_webarena/requirements.txt\n"
                "Then run Playwright browser setup if needed:\n"
                "  playwright install chromium"
            ) from exc
        raise SystemExit(
            f"BrowserGym benchmark module {module_name!r} is installed, but an "
            f"internal dependency is missing: {missing_name!r}.\n"
            "Rebuild the WebArena environment with:\n"
            "  bash examples/scripts/setup_benchmark_env.sh webarena"
        ) from exc


def build_prompt(
    *,
    goal: str,
    obs: dict[str, Any],
    obs_mode: str,
    obs_max_chars: int,
    action_description: str,
    previous_steps: list[dict[str, str]],
) -> list[dict[str, str]]:
    history = "\n".join(
        f"Step {item['step']}: action={item['action']} error={item['error']}"
        for item in previous_steps[-5:]
    )
    user = f"""Task:
{goal}

Recent steps:
{history or "None"}

Current observation:
{observation_text(obs, mode=obs_mode, max_chars=obs_max_chars)}

Available actions:
{action_description}

Choose the next browser action. Respond with exactly:
Thought: <brief reason>
Action: <one BrowserGym action>
"""
    return [
        {
            "role": "system",
            "content": (
                "You are a web-browsing agent running inside BrowserGym. "
                "Use only the listed BrowserGym actions. When the task is done, "
                "use send_msg_to_user('<final answer>')."
            ),
        },
        {"role": "user", "content": user},
    ]


def run_episode(
    *,
    args: argparse.Namespace,
    client: OpenAI,
    action_set: HighLevelActionSet,
    task_id: str,
) -> EpisodeResult:
    env_id = resolve_env_id(args.env_prefix, task_id)
    trace_path = Path(args.output_dir) / "traces" / f"{task_id.replace('/', '_')}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    llm_kwargs = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_output_tokens,
    }

    benchmark = args.benchmark
    framework = DEFAULT_FRAMEWORK
    program_id = ""
    success = False
    reward = 0.0
    step_count = 0
    error: str | None = None

    with browsergym_instance(
        llm_kwargs,
        instance_id=task_id,
        base_url=args.base_url,
        scaffold=DEFAULT_SCAFFOLD,
    ) as (program, patched_llm_kwargs):
        program_id = program.program_id
        env = gym.make(
            env_id,
            headless=args.headless,
            action_mapping=action_set.to_python_code,
        )
        try:
            reset_start = now()
            obs, reset_info = env.reset(seed=args.seed)
            reset_end = now()
            obs_sizes = observation_size(obs)
            write_jsonl(
                trace_path,
                trace_event(
                    task_id=task_id,
                    benchmark=benchmark,
                    framework=framework,
                    model=args.model,
                    step_id=0,
                    event_type="env_observation",
                    start=reset_start,
                    end=reset_end,
                    observation_size_bytes=obs_sizes["total"],
                    state_delta={
                        "phase": "reset",
                        "env_id": env_id,
                        "program_id": program_id,
                        "obs_sizes": obs_sizes,
                        "task_info": reset_info.get("task_info", {}),
                    },
                ),
            )

            goal = extract_goal(obs)
            previous_steps: list[dict[str, str]] = []
            terminated = False
            truncated = False

            for step_id in range(1, args.max_steps + 1):
                messages = build_prompt(
                    goal=goal,
                    obs=obs,
                    obs_mode=args.observation_mode,
                    obs_max_chars=args.observation_max_chars,
                    action_description=action_set.describe(
                        with_long_description=not args.compact_actions,
                        with_examples=not args.compact_actions,
                    ),
                    previous_steps=previous_steps,
                )
                llm_start = now()
                response = client.chat.completions.create(
                    messages=messages,
                    extra_body=patched_llm_kwargs.get("extra_body"),
                    **{
                        key: value
                        for key, value in patched_llm_kwargs.items()
                        if key != "extra_body"
                    },
                )
                llm_end = now()

                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

                action = parse_action(content, set(action_set.action_set))
                tool_name = first_action_name(action)

                write_jsonl(
                    trace_path,
                    trace_event(
                        task_id=task_id,
                        benchmark=benchmark,
                        framework=framework,
                        model=args.model,
                        step_id=step_id,
                        event_type="llm_call",
                        start=llm_start,
                        end=llm_end,
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        context_tokens=total_tokens or prompt_tokens + completion_tokens,
                        state_delta={
                            "program_id": program_id,
                            "raw_response": content,
                            "parsed_action": action,
                        },
                    ),
                )

                step_start = now()
                obs, reward, terminated, truncated, info = env.step(action)
                step_end = now()
                action_start = float(info.get("action_exec_start", step_start))
                action_end = float(info.get("action_exec_stop", step_end))
                action_error = obs.get("last_action_error") or None

                write_jsonl(
                    trace_path,
                    trace_event(
                        task_id=task_id,
                        benchmark=benchmark,
                        framework=framework,
                        model=args.model,
                        step_id=step_id,
                        event_type="tool_call",
                        start=action_start,
                        end=action_end,
                        tool_name=tool_name,
                        tool_args_size=len(action.encode("utf-8")),
                        success=not bool(action_error),
                        error_type=action_error.split(":", 1)[0] if action_error else None,
                        state_delta={
                            "program_id": program_id,
                            "action": action,
                            "action_error": action_error,
                            "action_exec_timeout_s": info.get("action_exec_timeout", 0),
                        },
                    ),
                )

                obs_sizes = observation_size(obs)
                write_jsonl(
                    trace_path,
                    trace_event(
                        task_id=task_id,
                        benchmark=benchmark,
                        framework=framework,
                        model=args.model,
                        step_id=step_id,
                        event_type="env_observation",
                        start=action_end,
                        end=step_end,
                        observation_size_bytes=obs_sizes["total"],
                        success=not bool(action_error),
                        error_type=action_error.split(":", 1)[0] if action_error else None,
                        state_delta={
                            "program_id": program_id,
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "obs_sizes": obs_sizes,
                            "task_info": info.get("task_info", {}),
                        },
                    ),
                )

                previous_steps.append(
                    {
                        "step": str(step_id),
                        "action": action,
                        "error": str(action_error or ""),
                    }
                )
                step_count = step_id
                if terminated or truncated:
                    break

            success = bool(reward)
            if not success and not (terminated or truncated):
                error = "max_steps_reached"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            write_jsonl(
                trace_path,
                trace_event(
                    task_id=task_id,
                    benchmark=benchmark,
                    framework=framework,
                    model=args.model,
                    step_id=step_count,
                    event_type="env_observation",
                    start=now(),
                    end=now(),
                    success=False,
                    error_type=type(exc).__name__,
                    state_delta={"program_id": program_id, "error": str(exc)},
                ),
            )
        finally:
            env.close()

    return EpisodeResult(
        task_id=task_id,
        env_id=env_id,
        program_id=program_id,
        success=success,
        reward=float(reward or 0.0),
        steps=step_count,
        output_path=str(trace_path),
        error=error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="webarena_verified")
    parser.add_argument("--benchmark-module", default="browsergym.webarena_verified")
    parser.add_argument("--env-prefix", default="browsergym/webarena_verified")
    parser.add_argument("--task-ids", default="0", help="Comma-separated task IDs or full env IDs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-subset", default="webarena")
    parser.add_argument("--observation-mode", choices=("axtree", "dom", "both"), default="axtree")
    parser.add_argument("--observation-max-chars", type=int, default=24000)
    parser.add_argument("--compact-actions", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    import_benchmark_module(args.benchmark_module)

    action_set = HighLevelActionSet(
        subsets=[args.action_subset],
        multiaction=False,
        strict=False,
    )
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]

    results = [
        run_episode(args=args, client=client, action_set=action_set, task_id=task_id)
        for task_id in task_ids
    ]

    results_path = Path(args.output_dir) / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump([result.__dict__ for result in results], handle, indent=2)

    print(json.dumps([result.__dict__ for result in results], indent=2))
    return 0 if all(result.error is None for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
