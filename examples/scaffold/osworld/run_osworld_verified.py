#!/usr/bin/env python3
"""Run OSWorld's native Qwen3VL multi-env runner through ThunderAgent.

The experiment loop, multiprocessing queue, DesktopEnv lifecycle, trajectory
logging, and evaluation are delegated to OSWorld's own
``scripts/python/run_multienv_qwen3vl.py``.  This wrapper only:

1. selects an optional task subset before invoking the native runner;
2. forces OSWorld's Qwen3VL agent to use an OpenAI-compatible endpoint; and
3. attaches one ThunderAgent ``program_id`` to each OSWorld task's model calls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "qwen3.5-27B"
DEFAULT_BASE_URL = "http://127.0.0.1:9000/v1"
DEFAULT_SCAFFOLD = "osworld-native-qwen35"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
PROGRESS_EVENT_FILENAME = "_progress.jsonl"

MOUSE_ACTIONS = {
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
}

COMPUTER_USE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": (
            "Control an Ubuntu desktop GUI from screenshots. Use relative "
            "coordinates on a 0..999 grid unless the caller explicitly asks "
            "for absolute coordinates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The next GUI action to execute.",
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "scroll",
                        "wait",
                        "terminate",
                    ],
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required only when action=key.",
                },
                "text": {
                    "type": "string",
                    "description": "Required only when action=type.",
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Required for mouse actions: [x, y].",
                },
                "pixels": {
                    "type": "number",
                    "description": "Scroll amount for action=scroll.",
                },
                "time": {
                    "type": "number",
                    "description": "Seconds to wait for action=wait.",
                },
                "duration": {
                    "type": "number",
                    "description": "Drag duration for action=left_click_drag.",
                },
                "status": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "Completion status for action=terminate.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--osworld-root",
        default=os.environ.get("OSWORLD_SOURCE_ROOT", str(Path.home() / "OSWorld")),
        help="Path to an OSWorld checkout.",
    )
    parser.add_argument("--native-runner", default="")
    parser.add_argument("--test-config-base-dir", "--test_config_base_dir", default="")
    parser.add_argument(
        "--test-all-meta-path",
        "--test_all_meta_path",
        default="",
    )
    parser.add_argument("--domain", default="all")
    parser.add_argument("--task-ids", "--task_ids", default="")
    parser.add_argument("--task-start", "--task_start", type=int, default=0)
    parser.add_argument(
        "--num-tasks",
        "--num_tasks",
        type=int,
        default=1,
        help="0 means all selected tasks.",
    )
    parser.add_argument("--output-dir", "--result-dir", "--result_dir", dest="result_dir", required=True)
    parser.add_argument("--max-concurrency", "--num-envs", "--num_envs", dest="num_envs", type=int, default=1)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", "--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", "--api_key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.9)
    parser.add_argument("--max-output-tokens", "--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=32768)
    parser.add_argument("--extra-body", "--extra_body", default="")
    parser.add_argument("--scaffold", default=DEFAULT_SCAFFOLD)

    # Qwen3VL-compatible names are kept because the native OSWorld agent is still
    # the Qwen3VL computer-use scaffold; the served model can be Qwen3.5.
    parser.add_argument("--qwen3vl-model", "--qwen35-model", dest="qwen_model", default="")
    parser.add_argument("--qwen3vl-base-url", "--qwen35-base-url", dest="qwen_base_url", default="")
    parser.add_argument("--qwen3vl-api-key", "--qwen35-api-key", dest="qwen_api_key", default="")
    parser.add_argument("--qwen3vl-max-tokens", "--qwen35-max-tokens", dest="qwen_max_tokens", type=int, default=0)
    parser.add_argument(
        "--qwen3vl-coordinate-type",
        "--qwen35-coordinate-type",
        "--coord",
        dest="coordinate_type",
        choices=["absolute", "relative"],
        default="relative",
    )
    parser.add_argument("--qwen3vl-history-n", "--qwen35-history-n", dest="history_n", type=int, default=4)
    parser.add_argument("--qwen3vl-request-timeout", "--qwen35-request-timeout", dest="request_timeout", type=float, default=500.0)
    parser.add_argument("--qwen3vl-max-retries", "--qwen35-max-retries", dest="max_retries", type=int, default=5)
    parser.add_argument("--qwen3vl-thinking-budget", "--qwen35-thinking-budget", dest="thinking_budget", type=int, default=32768)
    parser.add_argument("--qwen3vl-enable-thinking", "--qwen35-enable-thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--qwen3vl-disable-thinking", "--qwen35-disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=False)
    parser.add_argument("--add-thought-prefix", "--add_thought_prefix", action="store_true")
    parser.add_argument(
        "--qwen3vl-use-vllm-tool-calls",
        "--qwen35-use-vllm-tool-calls",
        dest="use_vllm_tool_calls",
        action=argparse.BooleanOptionalAction,
        default=env_flag("QWEN3VL_USE_VLLM_TOOL_CALLS", False),
        help="Send OpenAI tools and consume message.tool_calls from vLLM.",
    )
    parser.add_argument(
        "--qwen3vl-tool-choice",
        "--qwen35-tool-choice",
        dest="tool_choice",
        choices=["auto", "required", "named"],
        default=os.environ.get("QWEN3VL_TOOL_CHOICE", "named"),
        help="Tool choice used when native vLLM tool calls are enabled.",
    )

    parser.add_argument("--path-to-vm", "--path_to_vm", dest="path_to_vm", default=None)
    parser.add_argument("--provider-name", "--provider_name", dest="provider_name", default="docker")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-width", "--screen_width", dest="screen_width", type=int, default=1920)
    parser.add_argument("--screen-height", "--screen_height", dest="screen_height", type=int, default=1080)
    parser.add_argument("--client-password", "--client_password", dest="client_password", default="")
    parser.add_argument("--action-space", "--action_space", dest="action_space", default="pyautogui")
    parser.add_argument(
        "--observation-type",
        "--observation_type",
        dest="observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
    )
    parser.add_argument("--sleep-after-execution", "--sleep_after_execution", dest="sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=15)
    parser.add_argument("--log-level", "--log_level", dest="log_level", default="INFO")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=env_flag("OSWORLD_PROGRESS", True),
        help="Show an overall OSWorld task progress bar while the native runner works.",
    )
    parser.add_argument(
        "--progress-interval",
        "--progress_interval",
        dest="progress_interval",
        type=float,
        default=env_float("OSWORLD_PROGRESS_INTERVAL", 10.0),
        help="Seconds between progress refreshes.",
    )

    # Compatibility knobs accepted by the older ThunderAgent-local runner.  The
    # native OSWorld Qwen3VL runner hardcodes these behaviors.
    parser.add_argument("--agent-kind", "--agent_kind", default="qwen35")
    parser.add_argument("--benchmark", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--framework", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--password", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-name", "--snapshot_name", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--os-type", "--os_type", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--enable-proxy",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--reset-sleep", "--reset_sleep", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--settle-sleep", "--settle_sleep", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--no-recording", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    return parser.parse_args()


def configure_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.qwen_model:
        args.model = args.qwen_model
    if args.qwen_base_url:
        args.base_url = args.qwen_base_url
    if args.qwen_api_key:
        args.api_key = args.qwen_api_key
    if args.qwen_max_tokens:
        args.max_tokens = args.qwen_max_tokens

    args.osworld_root = str(Path(args.osworld_root).expanduser().resolve())
    osworld_root = Path(args.osworld_root)
    if not args.test_config_base_dir:
        args.test_config_base_dir = str(osworld_root / "evaluation_examples")
    if not args.test_all_meta_path:
        args.test_all_meta_path = str(osworld_root / "evaluation_examples" / "test_nogdrive.json")
    args.result_dir = str(Path(args.result_dir).expanduser().resolve())

    if args.task_start < 0:
        raise ValueError("--task-start must be >= 0")
    if args.num_tasks < 0:
        raise ValueError("--num-tasks must be >= 0")
    if args.num_envs < 1:
        raise ValueError("--max-concurrency/--num-envs must be >= 1")
    if args.history_n < 0:
        raise ValueError("--qwen3vl-history-n must be >= 0")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be > 0")
    if args.max_retries < 1:
        raise ValueError("--qwen3vl-max-retries must be >= 1")
    if args.action_space != "pyautogui":
        raise ValueError("OSWorld Qwen3VL runner supports only pyautogui action space")
    if args.observation_type != "screenshot":
        raise ValueError("OSWorld Qwen3VL runner supports only screenshot observation type")

    safe_json_object(args.extra_body, option_name="--extra-body")
    return args


def safe_json_object(value: str, *, option_name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return parsed


def native_tool_system_prompt(coordinate_type: str) -> str:
    if coordinate_type == "absolute":
        coordinate_rule = "Use actual screenshot pixel coordinates for mouse actions."
    else:
        coordinate_rule = "Use relative coordinates on a 0..999 by 0..999 screen grid."
    return "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer GUI from screenshots.",
            coordinate_rule,
            "For every step, call exactly one computer_use tool, including task completion.",
            "Choose the smallest reliable UI action; wait when the screenshot has not updated yet.",
            "When the task is complete, call computer_use with action=terminate and status=success.",
            'Do not use the legacy JSON wrapper {"name": ..., "arguments": ...}; use the provided tool call interface.',
            "If you include text before the tool call, keep it to one short Action line.",
        ]
    )


def native_tool_choice(value: str) -> str | dict[str, Any]:
    if value == "named":
        return {"type": "function", "function": {"name": "computer_use"}}
    return value


def clone_content(content: Any) -> Any:
    if isinstance(content, list):
        return [dict(part) if isinstance(part, dict) else part for part in content]
    if isinstance(content, dict):
        return dict(content)
    return content


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
                elif "text" in part:
                    chunks.append(str(part.get("text", "")))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def message_payload_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"message_count": len(messages), "image_count": 0, "text_chars": 0}

    def visit(content: Any) -> None:
        if isinstance(content, str):
            stats["text_chars"] += len(content)
        elif isinstance(content, list):
            for part in content:
                visit(part)
        elif isinstance(content, dict):
            part_type = content.get("type")
            if part_type in {"image_url", "input_image"} or "image_url" in content:
                stats["image_count"] += 1
                return
            if "text" in content:
                stats["text_chars"] += len(str(content.get("text") or ""))
                return
            for value in content.values():
                visit(value)

    for message in messages:
        visit(message.get("content"))
    return stats


def object_value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def completion_usage_metadata(usage: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = object_value(usage, name)
        if value is not None:
            metadata[name] = value

    prompt_details = object_value(usage, "prompt_tokens_details")
    cached_tokens = object_value(prompt_details, "cached_tokens")
    if cached_tokens is not None:
        metadata["cached_tokens"] = cached_tokens

    completion_details = object_value(usage, "completion_tokens_details")
    reasoning_tokens = object_value(completion_details, "reasoning_tokens")
    if reasoning_tokens is not None:
        metadata["reasoning_tokens"] = reasoning_tokens
    return metadata


def choice_finish_reason(choice: Any) -> Any:
    return object_value(choice, "finish_reason")


def count_items(value: Any) -> int | None:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return None


def bytes_len(value: Any) -> int | None:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    return None


def summarize_text(value: Any, *, limit: int = 240) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def first_action_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or (stripped.startswith("<") and stripped.endswith(">")):
            continue
        if stripped.lower().startswith("action:"):
            return stripped.split(":", 1)[1].strip()
        return stripped
    return ""


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in '[{"-0123456789tfn':
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def parse_numeric(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    try:
        parsed = float(stripped)
    except ValueError:
        return value
    return int(parsed) if parsed.is_integer() else parsed


def parse_tool_parameter_value(name: str, raw_value: str) -> Any:
    value = raw_value
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    if name in {"coordinate", "keys"}:
        return parse_jsonish(value.strip())
    if name in {"pixels", "time", "duration"}:
        return parse_numeric(value.strip())
    if name in {"action", "status"}:
        return value.strip()
    return value


def normalize_tool_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, str):
        parsed = parse_jsonish(raw_args)
        args = parsed if isinstance(parsed, dict) else {"text": raw_args}
    elif isinstance(raw_args, dict):
        args = dict(raw_args)
    else:
        args = {}

    if isinstance(args.get("parameters"), dict) and "action" not in args:
        args = dict(args["parameters"])

    action = args.get("action")
    nested = args.get("arguments")
    if isinstance(nested, dict):
        args.pop("arguments", None)
        args.update(nested)
    elif isinstance(nested, list):
        args.pop("arguments", None)
        if action in MOUSE_ACTIONS and "coordinate" not in args:
            args["coordinate"] = nested
    elif isinstance(nested, str):
        args.pop("arguments", None)
        parsed_nested = parse_jsonish(nested)
        if action in MOUSE_ACTIONS and "coordinate" not in args:
            args["coordinate"] = parsed_nested
        elif action == "key" and "keys" not in args:
            args["keys"] = parsed_nested
        elif action == "type" and "text" not in args:
            args["text"] = nested
        elif action == "scroll" and "pixels" not in args:
            args["pixels"] = parse_numeric(nested)
        elif action == "wait" and "time" not in args:
            args["time"] = parse_numeric(nested)
        elif action == "terminate" and "status" not in args:
            args["status"] = nested.strip()

    if "coordinate" in args:
        args["coordinate"] = parse_jsonish(args["coordinate"])
    if "keys" in args:
        keys = parse_jsonish(args["keys"])
        args["keys"] = keys if isinstance(keys, list) else [str(keys)]
    for numeric_name in ("pixels", "time", "duration"):
        if numeric_name in args:
            args[numeric_name] = parse_numeric(args[numeric_name])

    return {key: value for key, value in args.items() if value is not None}


def format_parameter_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def tool_call_to_qwen_xml(name: str, args: dict[str, Any]) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in args.items():
        lines.extend(
            [
                f"<parameter={key}>",
                format_parameter_value(value),
                "</parameter>",
            ]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def extract_legacy_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "arguments" in payload:
            calls.append(payload)
    if calls:
        return calls
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return calls
        if isinstance(payload, dict) and "arguments" in payload:
            calls.append(payload)
    return calls


def legacy_tool_response_to_qwen_xml(text: str) -> str:
    calls = extract_legacy_tool_calls(text)
    if not calls:
        return text
    parts: list[str] = []
    action_line = first_action_line(text)
    if action_line:
        parts.append(f"Action: {action_line}")
    for call in calls:
        name = str(call.get("name") or "computer_use")
        args = normalize_tool_arguments(call.get("arguments", {}))
        parts.append(tool_call_to_qwen_xml(name, args))
    return "\n".join(parts)


def parse_qwen_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for function_match in re.finditer(
        r"<function=([^>\n]+)>\s*(.*?)\s*</function>", text, re.DOTALL
    ):
        name = function_match.group(1).strip()
        body = function_match.group(2)
        args: dict[str, Any] = {}
        for param_match in re.finditer(
            r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            body,
            re.DOTALL,
        ):
            param_name = param_match.group(1).strip()
            args[param_name] = parse_tool_parameter_value(
                param_name, param_match.group(2)
            )
        calls.append({"name": name, "arguments": normalize_tool_arguments(args)})
    return calls


def tool_call_object_parts(tool_call: Any) -> tuple[str, Any]:
    function = getattr(tool_call, "function", None)
    if function is None and isinstance(tool_call, dict):
        function = tool_call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or ""), function.get("arguments")
    return str(getattr(function, "name", "") or ""), getattr(function, "arguments", None)


def legacy_response_from_calls(calls: list[dict[str, Any]], content: str = "") -> str:
    blocks: list[str] = []
    first_action = ""
    for call in calls:
        name = str(call.get("name") or "computer_use")
        args = normalize_tool_arguments(call.get("arguments", {}))
        if not args.get("action"):
            continue
        if not first_action:
            first_action = str(args.get("action"))
        payload = {"name": name, "arguments": args}
        blocks.append(
            "<tool_call>\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\n</tool_call>"
        )
    if not blocks:
        return ""
    action_line = first_action_line(content) or first_action
    return "\n".join([f"Action: {action_line}", *blocks])


def tool_calls_to_legacy_response(tool_calls: Any, content: str = "") -> str:
    calls: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        name, raw_args = tool_call_object_parts(tool_call)
        if not name:
            continue
        calls.append({"name": name, "arguments": normalize_tool_arguments(raw_args)})
    return legacy_response_from_calls(calls, content)


def qwen_xml_content_to_legacy_response(content: str) -> str:
    return legacy_response_from_calls(parse_qwen_xml_tool_calls(content), content)


def prepare_messages_for_native_tool_calls(
    messages: list[dict[str, Any]], coordinate_type: str
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    saw_system = False
    for message in messages:
        copied = {key: clone_content(value) for key, value in message.items()}
        role = copied.get("role")
        if role == "system":
            copied["content"] = text_content(native_tool_system_prompt(coordinate_type))
            saw_system = True
        elif role == "assistant":
            copied["content"] = text_content(
                legacy_tool_response_to_qwen_xml(content_to_text(copied.get("content")))
            )
        prepared.append(copied)
    if not saw_system:
        prepared.insert(
            0,
            {
                "role": "system",
                "content": text_content(native_tool_system_prompt(coordinate_type)),
            },
        )
    return prepared


def load_meta(path: str) -> dict[str, list[str]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(domain): [str(example_id) for example_id in examples] for domain, examples in payload.items()}


def select_tasks(args: argparse.Namespace) -> dict[str, list[str]]:
    meta = load_meta(args.test_all_meta_path)
    selected: list[tuple[str, str]] = []

    if args.task_ids:
        raw_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
        for raw_id in raw_ids:
            if "/" in raw_id:
                domain, example_id = raw_id.split("/", 1)
            elif args.domain != "all":
                domain, example_id = args.domain, raw_id
            else:
                matches = [
                    (domain_name, raw_id)
                    for domain_name, examples in meta.items()
                    if raw_id in examples
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Task id {raw_id!r} is ambiguous or missing; use domain/example_id."
                    )
                domain, example_id = matches[0]
            if domain not in meta or example_id not in meta[domain]:
                raise ValueError(f"Unknown OSWorld task: {domain}/{example_id}")
            selected.append((domain, example_id))
    else:
        for domain, examples in meta.items():
            if args.domain == "all" or args.domain == domain:
                selected.extend((domain, example_id) for example_id in examples)
        if args.task_start:
            selected = selected[args.task_start :]
        if args.num_tasks:
            selected = selected[: args.num_tasks]

    if not selected:
        raise ValueError("No OSWorld tasks selected.")

    output: dict[str, list[str]] = {}
    for domain, example_id in selected:
        output.setdefault(domain, []).append(example_id)
    return output


def write_selected_meta(result_dir: str, selected_meta: dict[str, list[str]]) -> str:
    path = Path(result_dir) / "_selected_test_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(selected_meta, handle, indent=2)
    return str(path)


def total_task_count(meta: dict[str, list[str]]) -> int:
    return sum(len(example_ids) for example_ids in meta.values())


def progress_event_path(result_dir: str | Path) -> Path:
    return Path(result_dir) / PROGRESS_EVENT_FILENAME


def reset_progress_events(result_dir: str | Path) -> Path:
    path = progress_event_path(result_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def write_progress_event(result_dir: str | Path, event: dict[str, Any]) -> None:
    path = progress_event_path(result_dir)
    payload = {
        "ts": time.time(),
        "pid": os.getpid(),
        **event,
    }
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    encoded = line.encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)
    except OSError:
        return


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(done: int, total: int, *, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = min(width, int(round(width * done / total)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


class ProgressMonitor:
    """Render a coarse overall progress bar from worker-written JSONL events."""

    def __init__(self, *, path: Path, total: int, interval_s: float) -> None:
        self.path = path
        self.total = total
        self.interval_s = interval_s
        self.started: set[str] = set()
        self.finished: dict[str, str] = {}
        self.step_count = 0
        self.last_task = ""
        self.last_action = ""
        self.start_time = time.time()
        self._offset = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="osworld-progress",
            daemon=True,
        )
        self._thread.start()
        self.render(force=True)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, min(self.interval_s, 5.0)))
        self.read_new_events()
        self.render(force=True, final=True)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.read_new_events()
            self.render()

    def read_new_events(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < self._offset:
            self._offset = 0
        try:
            with self.path.open(encoding="utf-8") as handle:
                handle.seek(self._offset)
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.apply_event(event)
                self._offset = handle.tell()
        except OSError:
            return

    def apply_event(self, event: dict[str, Any]) -> None:
        task_id = str(event.get("task_id") or "")
        if task_id:
            self.last_task = task_id
        event_name = event.get("event")
        if event_name == "task_start" and task_id:
            self.started.add(task_id)
        elif event_name == "task_step":
            self.step_count += 1
            action = event.get("action")
            if action:
                self.last_action = summarize_text(action, limit=80)
        elif event_name == "task_end" and task_id:
            self.finished[task_id] = str(event.get("status") or "done")

    def render(self, *, force: bool = False, final: bool = False) -> None:
        del force
        done = len(self.finished)
        active = len(self.started - set(self.finished))
        errors = sum(1 for status in self.finished.values() if status != "done")
        pct = (done / self.total * 100.0) if self.total else 0.0
        line = (
            f"[OSWorld progress] {progress_bar(done, self.total)} "
            f"{done}/{self.total} tasks ({pct:5.1f}%) "
            f"active={active} steps={self.step_count} errors={errors} "
            f"elapsed={format_duration(time.time() - self.start_time)}"
        )
        if self.last_task:
            line += f" last={self.last_task}"
        if self.last_action and not final:
            line += f" action={self.last_action}"
        if final:
            line += " final"
        print(line, file=sys.stderr, flush=True)


def should_rewrite_hf_urls() -> bool:
    return os.environ.get("OSWORLD_REWRITE_HF_URLS", "1") != "0"


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).strip().rstrip("/")


def rewrite_hf_url(value: str) -> str:
    endpoint = hf_endpoint()
    if not endpoint or "huggingface.co" not in value:
        return value

    parsed = urlsplit(value)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        return value

    endpoint_parts = urlsplit(endpoint)
    if not endpoint_parts.scheme or not endpoint_parts.netloc:
        return value

    return urlunsplit(
        (
            endpoint_parts.scheme,
            endpoint_parts.netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def rewrite_hf_urls(value: Any) -> Any:
    if isinstance(value, str):
        return rewrite_hf_url(value)
    if isinstance(value, list):
        return [rewrite_hf_urls(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_hf_urls(item) for key, item in value.items()}
    return value


def install_hf_url_rewrite_patch() -> None:
    if not should_rewrite_hf_urls():
        return

    import requests

    session_cls = requests.sessions.Session
    if getattr(session_cls, "_thunderagent_hf_rewrite_patched", False):
        return

    original_request = session_cls.request

    def patched_request(self: Any, method: str, url: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(url, str):
            url = rewrite_hf_url(url)
        return original_request(self, method, url, *args, **kwargs)

    session_cls.request = patched_request
    session_cls._thunderagent_hf_rewrite_patched = True


def ensure_import_paths(osworld_root: str) -> None:
    for path in (str(REPO_ROOT), osworld_root):
        if path not in sys.path:
            sys.path.insert(0, path)


def install_thunderagent_patches(
    args: argparse.Namespace,
    *,
    id_to_domain: dict[str, str],
) -> None:
    import lib_run_single
    from examples.adapters.osworld import osworld_instance
    from examples.scaffold.osworld.timing_trace import (
        TaskTrace,
        current_trace,
        maybe_span,
        reset_current_trace,
        set_current_trace,
    )
    from ThunderAgent.adapters import inject_program_id
    import desktop_env.desktop_env as desktop_env_module
    from desktop_env.controllers.python import PythonController
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents import qwen3vl_agent as qwen3vl_module

    parsed_extra_body = safe_json_object(args.extra_body, option_name="--extra-body")
    qwen3vl_module.MAX_RETRY_TIMES = args.max_retries

    if not getattr(qwen3vl_module.Qwen3VLAgent, "_thunderagent_predict_timing_patched", False):
        original_predict = qwen3vl_module.Qwen3VLAgent.predict

        def patched_predict(self: Any, instruction: str, obs: dict[str, Any]) -> Any:
            step_id = len(getattr(self, "actions", [])) + 1
            screenshot_size = bytes_len(obs.get("screenshot")) if isinstance(obs, dict) else None
            trace = current_trace()
            with maybe_span(
                trace,
                "agent.predict",
                step_id=step_id,
                metadata={
                    "instruction_chars": len(instruction or ""),
                    "screenshot_bytes": screenshot_size,
                    "history_actions": len(getattr(self, "actions", [])),
                },
            ) as span:
                response, actions = original_predict(self, instruction, obs)
                if isinstance(actions, list):
                    action_count = len(actions)
                elif actions is None:
                    action_count = 0
                else:
                    action_count = 1
                span.update(
                    action_count=action_count,
                    response_len=len(response or ""),
                )
                return response, actions

        qwen3vl_module.Qwen3VLAgent.predict = patched_predict
        qwen3vl_module.Qwen3VLAgent._thunderagent_predict_timing_patched = True

    if not getattr(qwen3vl_module.Qwen3VLAgent, "_thunderagent_parse_timing_patched", False):
        original_parse_response = qwen3vl_module.Qwen3VLAgent.parse_response

        def patched_parse_response(
            self: Any,
            response: str,
            *parse_args: Any,
            **parse_kwargs: Any,
        ) -> Any:
            step_id = len(getattr(self, "actions", [])) + 1
            trace = current_trace()
            with maybe_span(
                trace,
                "parse_response_to_pyautogui",
                step_id=step_id,
                metadata={
                    "response_len": len(response or ""),
                    "coordinate_type": getattr(self, "coordinate_type", None),
                },
            ) as span:
                low_level_instruction, pyautogui_code = original_parse_response(
                    self, response, *parse_args, **parse_kwargs
                )
                first_action = (
                    pyautogui_code[0]
                    if isinstance(pyautogui_code, list) and pyautogui_code
                    else ""
                )
                span.update(
                    low_level_action=summarize_text(low_level_instruction),
                    pyautogui_action_count=len(pyautogui_code)
                    if isinstance(pyautogui_code, list)
                    else 0,
                    first_pyautogui_action=summarize_text(first_action),
                )
                return low_level_instruction, pyautogui_code

        qwen3vl_module.Qwen3VLAgent.parse_response = patched_parse_response
        qwen3vl_module.Qwen3VLAgent._thunderagent_parse_timing_patched = True

    if not getattr(DesktopEnv, "_thunderagent_step_timing_patched", False):
        def patched_step(self: Any, action: Any, pause: float = 2) -> Any:
            raw_step_no = getattr(self, "_step_no", 0)
            try:
                step_id = int(raw_step_no) + 1
            except (TypeError, ValueError):
                step_id = None
            trace = current_trace()
            trace_task_id = getattr(trace, "task_id", "") if trace is not None else ""
            if trace_task_id:
                write_progress_event(
                    args.result_dir,
                    {
                        "event": "task_step",
                        "task_id": trace_task_id,
                        "step_id": step_id,
                        "action": summarize_text(action),
                    },
                )
            with maybe_span(
                trace,
                "desktop_env.step",
                step_id=step_id,
                metadata={
                    "action_type": type(action).__name__,
                    "action_summary": summarize_text(action),
                    "pause_s": pause,
                },
            ) as span:
                self._step_no += 1
                self.action_history.append(action)
                self.is_environment_used = True

                reward = 0
                done = False
                info: dict[str, Any] = {}
                desktop_env_module.logger.info(
                    "Step %s in trajectory %s with action: %s",
                    self._step_no,
                    self._traj_no,
                    action,
                )

                is_special_action = action in ["WAIT", "FAIL", "DONE"] or (
                    isinstance(action, dict)
                    and action.get("action_type") in ["WAIT", "FAIL", "DONE"]
                )
                if is_special_action:
                    if action == "WAIT" or (
                        type(action) == dict and action.get("action_type") == "WAIT"
                    ):
                        with maybe_span(
                            trace,
                            "desktop_env.special_action_sleep",
                            metadata={"sleep_s": pause},
                        ):
                            time.sleep(pause)
                    elif action == "FAIL" or (
                        type(action) == dict and action.get("action_type") == "FAIL"
                    ):
                        done = True
                        info = {"fail": True}
                    elif action == "DONE" or (
                        type(action) == dict and action.get("action_type") == "DONE"
                    ):
                        done = True
                        info = {"done": True}

                if self.action_space == "computer_13":
                    self.controller.execute_action(action)
                elif self.action_space in {"pyautogui", "claude_computer_use"}:
                    if is_special_action:
                        self.controller.execute_action(action)
                    else:
                        if type(action) == str:
                            command = action
                            self.controller.execute_python_command(command)
                        elif type(action) == dict:
                            command = action["command"]
                            self.controller.execute_python_command(command)

                with maybe_span(
                    trace,
                    "desktop_env.sleep_after_execution",
                    metadata={"sleep_s": pause},
                ):
                    time.sleep(pause)
                observation = self._get_obs()

                span.update(reward=reward, done=done, info=info)
                return observation, reward, done, info

        DesktopEnv.step = patched_step
        DesktopEnv._thunderagent_step_timing_patched = True

    if not getattr(DesktopEnv, "_thunderagent_get_obs_timing_patched", False):
        original_get_obs = DesktopEnv._get_obs

        def patched_get_obs(self: Any) -> Any:
            raw_step_no = getattr(self, "_step_no", None)
            step_id = raw_step_no if isinstance(raw_step_no, int) else None
            trace = current_trace()
            with maybe_span(
                trace,
                "desktop_env.get_obs",
                step_id=step_id,
                metadata={
                    "require_a11y_tree": getattr(self, "require_a11y_tree", None),
                    "require_terminal": getattr(self, "require_terminal", None),
                },
            ) as span:
                observation = original_get_obs(self)
                screenshot = (
                    observation.get("screenshot")
                    if isinstance(observation, dict)
                    else None
                )
                span.update(
                    screenshot_bytes=bytes_len(screenshot),
                    screenshot_success=screenshot is not None,
                    has_accessibility_tree=bool(
                        observation.get("accessibility_tree")
                    )
                    if isinstance(observation, dict)
                    else None,
                    has_terminal=bool(observation.get("terminal"))
                    if isinstance(observation, dict)
                    else None,
                )
                return observation

        DesktopEnv._get_obs = patched_get_obs
        DesktopEnv._thunderagent_get_obs_timing_patched = True

    if not getattr(DesktopEnv, "_thunderagent_evaluate_timing_patched", False):
        original_evaluate = DesktopEnv.evaluate

        def patched_evaluate(self: Any) -> Any:
            trace = current_trace()
            with maybe_span(
                trace,
                "desktop_env.evaluate",
                metadata={"task_id": getattr(self, "task_id", None)},
            ) as span:
                result = original_evaluate(self)
                span.update(result=result)
                return result

        DesktopEnv.evaluate = patched_evaluate
        DesktopEnv._thunderagent_evaluate_timing_patched = True

    if not getattr(PythonController, "_thunderagent_screenshot_timing_patched", False):
        original_get_screenshot = PythonController.get_screenshot

        def patched_get_screenshot(self: Any) -> Any:
            trace = current_trace()
            with maybe_span(trace, "controller.get_screenshot") as span:
                screenshot = original_get_screenshot(self)
                span.update(
                    success=screenshot is not None,
                    image_bytes=bytes_len(screenshot),
                )
                return screenshot

        PythonController.get_screenshot = patched_get_screenshot
        PythonController._thunderagent_screenshot_timing_patched = True

    if not getattr(PythonController, "_thunderagent_execute_timing_patched", False):
        original_execute_python_command = PythonController.execute_python_command

        def patched_execute_python_command(self: Any, command: str) -> Any:
            trace = current_trace()
            with maybe_span(
                trace,
                "controller.execute_python_command",
                metadata={
                    "command_summary": summarize_text(command),
                    "command_chars": len(command or ""),
                },
            ) as span:
                result = original_execute_python_command(self, command)
                returncode = None
                stdout_len = None
                stderr_len = None
                if isinstance(result, dict):
                    for key in ("returncode", "return_code", "exit_code", "code"):
                        if key in result:
                            returncode = result[key]
                            break
                    stdout_len = len(str(result.get("stdout", result.get("output", ""))))
                    stderr_len = len(str(result.get("stderr", "")))
                span.update(
                    success=result is not None,
                    returncode=returncode,
                    stdout_chars=stdout_len,
                    stderr_chars=stderr_len,
                )
                return result

        PythonController.execute_python_command = patched_execute_python_command
        PythonController._thunderagent_execute_timing_patched = True

    if not getattr(qwen3vl_module.Qwen3VLAgent, "_thunderagent_init_patched", False):
        original_init = qwen3vl_module.Qwen3VLAgent.__init__

        def patched_init(self, *init_args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("api_backend", "openai")
            kwargs.setdefault("history_n", args.history_n)
            kwargs.setdefault("enable_thinking", args.enable_thinking)
            kwargs.setdefault("thinking_budget", args.thinking_budget)
            original_init(self, *init_args, **kwargs)

        qwen3vl_module.Qwen3VLAgent.__init__ = patched_init
        qwen3vl_module.Qwen3VLAgent._thunderagent_init_patched = True

    if not getattr(qwen3vl_module.Qwen3VLAgent, "_thunderagent_reset_patched", False):
        original_reset = qwen3vl_module.Qwen3VLAgent.reset

        def patched_reset(self: Any, *reset_args: Any, **reset_kwargs: Any) -> Any:
            reset_kwargs.pop("vm_ip", None)
            return original_reset(self, *reset_args, **reset_kwargs)

        qwen3vl_module.Qwen3VLAgent.reset = patched_reset
        qwen3vl_module.Qwen3VLAgent._thunderagent_reset_patched = True

    def patched_call_llm_openai(self, messages: list[dict[str, Any]], model: str) -> str:
        import openai

        client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
        logger = qwen3vl_module.logger
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            step_id = len(getattr(self, "actions", [])) + 1
            trace = current_trace()
            if logger is not None:
                logger.info(
                    "[OpenAI/ThunderAgent] Generating content with model: %s "
                    "(attempt %s/%s)",
                    model,
                    attempt,
                    args.max_retries,
                )
            try:
                with maybe_span(
                    trace,
                    "llm.prepare_request",
                    step_id=step_id,
                    metadata={
                        "attempt": attempt,
                        "model": model,
                        "use_vllm_tool_calls": bool(args.use_vllm_tool_calls),
                    },
                ) as prepare_span:
                    extra_body = dict(parsed_extra_body)
                    chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}
                    if not isinstance(chat_template_kwargs, dict):
                        chat_template_kwargs = {}
                    chat_template_kwargs["enable_thinking"] = bool(args.enable_thinking)
                    extra_body["chat_template_kwargs"] = chat_template_kwargs
                    if args.enable_thinking:
                        extra_body.setdefault(
                            "thinking_token_budget", args.thinking_budget
                        )
                    extra_body = inject_program_id(extra_body)
                    request_messages = messages
                    request_kwargs: dict[str, Any] = {}
                    if args.use_vllm_tool_calls:
                        request_messages = prepare_messages_for_native_tool_calls(
                            messages, args.coordinate_type
                        )
                        request_kwargs.update(
                            {
                                "tools": [COMPUTER_USE_TOOL],
                                "tool_choice": native_tool_choice(args.tool_choice),
                                "parallel_tool_calls": False,
                            }
                        )
                    request_stats = message_payload_stats(request_messages)
                    prepare_span.update(
                        **request_stats,
                        original_message_count=len(messages),
                        extra_body_keys=sorted(extra_body.keys()),
                        has_program_id=bool(extra_body.get("program_id")),
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )

                request_metadata: dict[str, Any] = {
                    "attempt": attempt,
                    "model": model,
                    "timeout_s": args.request_timeout,
                    "use_vllm_tool_calls": bool(args.use_vllm_tool_calls),
                    **request_stats,
                }
                with maybe_span(
                    trace,
                    "llm.request",
                    step_id=step_id,
                    metadata=request_metadata,
                ) as request_span:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=request_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        extra_body=extra_body,
                        timeout=args.request_timeout,
                        **request_kwargs,
                    )
                    choices = object_value(completion, "choices", []) or []
                    first_choice = choices[0] if choices else None
                    request_span.update(
                        choice_count=count_items(choices),
                        finish_reason=choice_finish_reason(first_choice),
                        response_id=object_value(completion, "id"),
                        **completion_usage_metadata(
                            object_value(completion, "usage")
                        ),
                    )

                choices = object_value(completion, "choices", []) or []
                message = object_value(choices[0], "message") if choices else None
                content = content_to_text(object_value(message, "content"))
                if args.use_vllm_tool_calls:
                    tool_calls = object_value(message, "tool_calls")
                    with maybe_span(
                        trace,
                        "llm.native_tool_calls_to_legacy",
                        step_id=step_id,
                        metadata={
                            "attempt": attempt,
                            "tool_call_count": count_items(tool_calls),
                            "content_chars": len(content),
                        },
                    ) as native_span:
                        native_response = tool_calls_to_legacy_response(
                            tool_calls, content
                        )
                        native_span.update(
                            parsed=bool(native_response),
                            legacy_response_chars=len(native_response),
                        )
                    if native_response:
                        if logger is not None:
                            logger.info(
                                "[OpenAI/ThunderAgent] Parsed native tool_calls from vLLM"
                            )
                        return native_response
                    with maybe_span(
                        trace,
                        "llm.xml_tool_call_parse",
                        step_id=step_id,
                        metadata={"attempt": attempt, "content_chars": len(content)},
                    ) as xml_span:
                        xml_response = qwen_xml_content_to_legacy_response(content)
                        xml_span.update(
                            parsed=bool(xml_response),
                            legacy_response_chars=len(xml_response),
                        )
                    if xml_response:
                        if logger is not None:
                            logger.info(
                                "[OpenAI/ThunderAgent] Parsed qwen XML tool call content"
                            )
                        return xml_response
                    if logger is not None:
                        logger.warning(
                            "[OpenAI/ThunderAgent] No native tool_calls returned; "
                            "falling back to content parser"
                        )
                return content
            except Exception as exc:  # match OSWorld's forgiving retry behavior
                last_error = exc
                if logger is not None:
                    logger.error("[OpenAI/ThunderAgent] Error calling model: %s", exc)
                if attempt < args.max_retries:
                    retry_sleep_s = min(5.0 * attempt, 30.0)
                    with maybe_span(
                        current_trace(),
                        "llm.retry_sleep",
                        step_id=step_id,
                        metadata={"attempt": attempt, "sleep_s": retry_sleep_s},
                    ):
                        time.sleep(retry_sleep_s)
        if logger is not None and last_error is not None:
            logger.error("[OpenAI/ThunderAgent] Exhausted retries: %s", last_error)
        return ""

    qwen3vl_module.Qwen3VLAgent._call_llm_openai = patched_call_llm_openai

    if not getattr(lib_run_single, "_thunderagent_run_single_patched", False):
        original_run_single_example = lib_run_single.run_single_example

        def patched_run_single_example(
            agent: Any,
            env: Any,
            example: dict[str, Any],
            max_steps: int,
            instruction: str,
            run_args: argparse.Namespace,
            example_result_dir: str,
            scores: Any,
        ) -> Any:
            example_id = str(example.get("id") or Path(example_result_dir).name)
            domain = id_to_domain.get(example_id) or Path(example_result_dir).parent.name
            patched_example = rewrite_hf_urls(example) if should_rewrite_hf_urls() else example
            instance_id = f"{domain}/{example_id}"
            write_progress_event(
                args.result_dir,
                {
                    "event": "task_start",
                    "task_id": instance_id,
                    "domain": domain,
                    "example_id": example_id,
                },
            )
            started_at = time.time()
            task_failed = False
            with osworld_instance(
                {},
                instance_id=instance_id,
                base_url=args.base_url,
                scaffold=args.scaffold,
            ) as (program, _):
                trace = TaskTrace(
                    task_id=instance_id,
                    program_id=program.program_id,
                    result_dir=example_result_dir,
                )
                trace_token = set_current_trace(trace)
                try:
                    with trace.span(
                        "osworld.task",
                        metadata={
                            "example_id": example_id,
                            "domain": domain,
                            "instruction_chars": len(instruction or ""),
                            "max_steps": max_steps,
                            "model": args.model,
                        },
                    ):
                        return original_run_single_example(
                            agent,
                            env,
                            patched_example,
                            max_steps,
                            instruction,
                            run_args,
                            example_result_dir,
                            scores,
                        )
                except Exception as exc:
                    task_failed = True
                    write_progress_event(
                        args.result_dir,
                        {
                            "event": "task_end",
                            "status": "error",
                            "task_id": instance_id,
                            "domain": domain,
                            "example_id": example_id,
                            "duration_s": time.time() - started_at,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    raise
                finally:
                    reset_current_trace(trace_token)
                    if not task_failed:
                        result_path = Path(example_result_dir) / "result.txt"
                        status = "done" if result_path.exists() else "missing_result"
                        result_value = None
                        if result_path.exists():
                            try:
                                result_value = float(
                                    result_path.read_text(encoding="utf-8").strip()
                                )
                            except (OSError, ValueError):
                                result_value = None
                        write_progress_event(
                            args.result_dir,
                            {
                                "event": "task_end",
                                "status": status,
                                "task_id": instance_id,
                                "domain": domain,
                                "example_id": example_id,
                                "duration_s": time.time() - started_at,
                                "result": result_value,
                            },
                        )

        lib_run_single.run_single_example = patched_run_single_example
        lib_run_single._thunderagent_run_single_patched = True


def build_native_argv(args: argparse.Namespace, selected_meta_path: str) -> list[str]:
    argv = [
        native_runner_path(args),
        "--action_space",
        args.action_space,
        "--observation_type",
        args.observation_type,
        "--sleep_after_execution",
        str(args.sleep_after_execution),
        "--max_steps",
        str(args.max_steps),
        "--test_config_base_dir",
        args.test_config_base_dir,
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_tokens",
        str(args.max_tokens),
        "--coord",
        args.coordinate_type,
        "--domain",
        "all",
        "--test_all_meta_path",
        selected_meta_path,
        "--result_dir",
        args.result_dir,
        "--num_envs",
        str(args.num_envs),
        "--log_level",
        args.log_level,
        "--region",
        args.region,
        "--provider_name",
        args.provider_name,
        "--client_password",
        args.client_password,
        "--screen_width",
        str(args.screen_width),
        "--screen_height",
        str(args.screen_height),
    ]
    if args.path_to_vm:
        argv.extend(["--path_to_vm", args.path_to_vm])
    if args.headless:
        argv.append("--headless")
    if args.add_thought_prefix:
        argv.append("--add_thought_prefix")
    return argv


def native_runner_path(args: argparse.Namespace) -> str:
    if args.native_runner:
        return str(Path(args.native_runner).expanduser().resolve())
    return str(Path(args.osworld_root) / "scripts" / "python" / "run_multienv_qwen3vl.py")


def main() -> int:
    args = configure_args(parse_args())
    osworld_root = Path(args.osworld_root)
    native_runner = Path(native_runner_path(args))
    if not osworld_root.exists():
        raise FileNotFoundError(f"OSWorld root not found: {osworld_root}")
    if not native_runner.exists():
        raise FileNotFoundError(f"OSWorld Qwen3VL runner not found: {native_runner}")
    if not Path(args.test_all_meta_path).exists():
        raise FileNotFoundError(f"OSWorld test meta not found: {args.test_all_meta_path}")

    selected_meta = select_tasks(args)
    selected_meta_path = write_selected_meta(args.result_dir, selected_meta)
    progress_path = reset_progress_events(args.result_dir)
    progress_monitor = (
        ProgressMonitor(
            path=progress_path,
            total=total_task_count(selected_meta),
            interval_s=args.progress_interval,
        )
        if args.progress
        else None
    )
    id_to_domain = {
        example_id: domain
        for domain, example_ids in selected_meta.items()
        for example_id in example_ids
    }

    ensure_import_paths(str(osworld_root))
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OSWORLD_SOURCE_ROOT"] = str(osworld_root)
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    install_hf_url_rewrite_patch()
    install_thunderagent_patches(args, id_to_domain=id_to_domain)

    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    try:
        (osworld_root / "logs").mkdir(exist_ok=True)
        os.chdir(osworld_root)
        sys.argv = build_native_argv(args, selected_meta_path)
        if progress_monitor is not None:
            progress_monitor.start()
        runpy.run_path(str(native_runner), run_name="__main__")
    finally:
        if progress_monitor is not None:
            progress_monitor.stop()
        sys.argv = old_argv
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
