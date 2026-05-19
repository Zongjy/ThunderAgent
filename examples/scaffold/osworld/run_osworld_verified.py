#!/usr/bin/env python3
"""Run OSWorld/OSWorld-Verified tasks through Agent-S and ThunderAgent.

OSWorld supplies the desktop environment, reset logic, task configs, and
evaluator. Agent-S supplies the GUI agent loop. ThunderAgent receives one
program_id per OSWorld task through OpenAI-compatible extra_body metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.adapters.osworld import osworld_instance


DOMAIN = "computer_use"
DEFAULT_BENCHMARK = "osworld-verified"
DEFAULT_FRAMEWORK = "agent-s-thunderagent"
DEFAULT_SCAFFOLD = "osworld-agent-s"
DEFAULT_AGENT_KIND = "agent-s"
DEFAULT_AGENT_FRAMEWORKS = {
    "agent-s": DEFAULT_FRAMEWORK,
    "opencua": "opencua-thunderagent",
}
DEFAULT_AGENT_SCAFFOLDS = {
    "agent-s": DEFAULT_SCAFFOLD,
    "opencua": "osworld-opencua",
}
PINNED_AGENT_S_VERSION = "0.3.2"


@dataclass(frozen=True)
class TaskSpec:
    domain: str
    example_id: str

    @property
    def instance_id(self) -> str:
        return f"{self.domain}/{self.example_id}"


@dataclass
class AgentStepPrediction:
    info: dict[str, Any]
    actions: list[str]
    latency_ms: float


@dataclass
class EpisodeResult:
    task_id: str
    domain: str
    example_id: str
    program_id: str
    score: float
    success: bool
    steps: int
    output_dir: str
    trace_path: str
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


def write_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


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
    program_id: str,
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
        "state_delta": {"program_id": program_id, **(state_delta or {})},
    }


def screenshot_to_png_bytes(screenshot: Any) -> bytes:
    if screenshot is None:
        return b""
    if isinstance(screenshot, bytes):
        return screenshot
    if isinstance(screenshot, bytearray):
        return bytes(screenshot)
    if isinstance(screenshot, str):
        path = Path(screenshot)
        if path.exists():
            return path.read_bytes()
        return screenshot.encode("utf-8", errors="replace")

    try:
        from PIL import Image
    except Exception as exc:
        raise TypeError(
            "Screenshot is not bytes/path and Pillow is unavailable."
        ) from exc

    if isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        try:
            import numpy as np

            image = Image.fromarray(np.asarray(screenshot))
        except Exception as exc:
            raise TypeError(f"Unsupported screenshot type: {type(screenshot)!r}") from exc

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def observation_size(obs: dict[str, Any]) -> dict[str, int]:
    screenshot_bytes = len(screenshot_to_png_bytes(obs.get("screenshot")))
    text_payload = {key: value for key, value in obs.items() if key != "screenshot"}
    return {
        "total": screenshot_bytes + json_size(text_payload),
        "screenshot": screenshot_bytes,
        "metadata": json_size(text_payload),
    }


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


def merge_mappings(*items: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(dict(item))
    return merged


def normalize_platform(os_type: str) -> str:
    lowered = os_type.lower()
    if "win" in lowered:
        return "windows"
    if "mac" in lowered or "darwin" in lowered:
        return "darwin"
    return "linux"


def normalize_agent_s_actions(actions: Any) -> list[str]:
    if actions is None:
        return []
    if isinstance(actions, str):
        candidates = [actions]
    else:
        try:
            candidates = list(actions)
        except TypeError:
            candidates = [str(actions)]
    normalized = [str(action).strip() for action in candidates if str(action).strip()]
    return normalized


def tool_name_from_action(action: str) -> str:
    if action in {"WAIT", "DONE", "FAIL"}:
        return action
    match = re.search(r"\bpyautogui\.(\w+)\s*\(", action)
    return match.group(1) if match else "pyautogui"


def validate_agent_s_version(strict: bool) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python 3.10+ in this project
        from importlib_metadata import PackageNotFoundError, version  # type: ignore

    try:
        installed = version("gui-agents")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Agent-S is not installed. Install the pinned dependency with: "
            "OSWORLD_SOURCE_ROOT=/path/to/OSWorld "
            "bash examples/scripts/setup_benchmark_env.sh osworld"
        ) from exc
    if strict and installed != PINNED_AGENT_S_VERSION:
        raise RuntimeError(
            f"Agent-S version mismatch: expected gui-agents=={PINNED_AGENT_S_VERSION}, "
            f"found {installed}."
        )
    return installed


def install_agent_s_thunderagent_patch() -> None:
    """Inject ThunderAgent program_id into Agent-S OpenAI-compatible engines."""
    import gui_agents.s3.core.engine as engine_mod
    from ThunderAgent.adapters import inject_program_id

    if getattr(engine_mod, "_thunderagent_program_id_patch", False):
        return

    original_openai_init = engine_mod.LMMEngineOpenAI.__init__
    original_openai_generate = engine_mod.LMMEngineOpenAI.generate

    def openai_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self._thunderagent_extra_body = dict(kwargs.pop("extra_body", {}) or {})
        self._thunderagent_max_new_tokens = kwargs.pop(
            "max_new_tokens", kwargs.pop("max_tokens", None)
        )
        self._thunderagent_top_p = kwargs.pop("top_p", None)
        original_openai_init(self, *args, **kwargs)

    def openai_generate(
        self: Any,
        messages: Any,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        request_extra_body = kwargs.pop("extra_body", None)
        extra_body = merge_mappings(
            getattr(self, "_thunderagent_extra_body", None),
            request_extra_body if isinstance(request_extra_body, Mapping) else None,
        )
        injected_extra_body = inject_program_id(extra_body)
        if injected_extra_body:
            kwargs["extra_body"] = injected_extra_body

        effective_max_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else getattr(self, "_thunderagent_max_new_tokens", None)
        )
        if (
            effective_max_tokens
            and "max_tokens" not in kwargs
            and "max_completion_tokens" not in kwargs
        ):
            kwargs["max_tokens"] = int(effective_max_tokens)

        top_p = getattr(self, "_thunderagent_top_p", None)
        if top_p is not None and "top_p" not in kwargs:
            kwargs["top_p"] = float(top_p)

        return original_openai_generate(
            self,
            messages,
            temperature=temperature,
            max_new_tokens=effective_max_tokens,
            **kwargs,
        )

    original_vllm_init = engine_mod.LMMEnginevLLM.__init__

    def vllm_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self._thunderagent_extra_body = dict(kwargs.pop("extra_body", {}) or {})
        self._thunderagent_max_new_tokens = kwargs.pop(
            "max_new_tokens", kwargs.pop("max_tokens", None)
        )
        self._thunderagent_top_p = kwargs.pop("top_p", None)
        original_vllm_init(self, *args, **kwargs)

    def vllm_generate(
        self: Any,
        messages: Any,
        temperature: float = 0.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        max_new_tokens: int | None = 512,
        **kwargs: Any,
    ) -> str:
        api_key = (
            self.api_key
            or os.getenv("vLLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        base_url = self.base_url or os.getenv("vLLM_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "A vLLM endpoint URL must be provided through base_url or "
                "vLLM_ENDPOINT_URL."
            )
        if not self.llm_client:
            self.llm_client = engine_mod.OpenAI(base_url=base_url, api_key=api_key)

        effective_top_p = getattr(self, "_thunderagent_top_p", None)
        if effective_top_p is None:
            effective_top_p = top_p
        configured_max_tokens = getattr(self, "_thunderagent_max_new_tokens", None)
        effective_max_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else configured_max_tokens
            if configured_max_tokens is not None
            else 4096
        )

        request_extra_body = kwargs.pop("extra_body", None)
        extra_body = merge_mappings(
            {"repetition_penalty": repetition_penalty},
            getattr(self, "_thunderagent_extra_body", None),
            request_extra_body if isinstance(request_extra_body, Mapping) else None,
        )
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=int(effective_max_tokens),
            temperature=self.temperature if self.temperature is not None else temperature,
            top_p=float(effective_top_p),
            extra_body=inject_program_id(extra_body),
            **kwargs,
        )
        return completion.choices[0].message.content

    engine_mod.LMMEngineOpenAI.__init__ = openai_init
    engine_mod.LMMEngineOpenAI.generate = openai_generate
    engine_mod.LMMEnginevLLM.__init__ = vllm_init
    engine_mod.LMMEnginevLLM.generate = vllm_generate
    engine_mod._thunderagent_program_id_patch = True


def agent_s_engine_params(
    args: argparse.Namespace,
    *,
    role: str,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prefix = f"agent_s_{role}_"
    engine_type = getattr(args, f"{prefix}engine_type") or args.agent_s_engine_type
    model = getattr(args, f"{prefix}model") or args.model
    base_url = getattr(args, f"{prefix}base_url") or args.base_url
    api_key = getattr(args, f"{prefix}api_key") or args.api_key
    role_extra_body = safe_json_object(
        getattr(args, f"{prefix}extra_body") or "",
        option_name=f"--agent-s-{role}-extra-body",
    )

    params: dict[str, Any] = {
        "engine_type": engine_type,
        "model": model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_output_tokens,
        "extra_body": merge_mappings(extra_body, role_extra_body),
    }
    if base_url:
        params["base_url"] = base_url
    if api_key:
        params["api_key"] = api_key

    if role == "grounding":
        params["grounding_width"] = args.agent_s_grounding_width
        params["grounding_height"] = args.agent_s_grounding_height

    return params


class AgentSOSWorldAgent:
    """Thin wrapper around Agent-S3 for one OSWorld episode."""

    def __init__(
        self,
        *,
        env: Any,
        args: argparse.Namespace,
        extra_body: Mapping[str, Any] | None,
    ) -> None:
        validate_agent_s_version(args.agent_s_strict_version)
        install_agent_s_thunderagent_patch()

        from gui_agents.s3.agents.agent_s import AgentS3
        from gui_agents.s3.agents.grounding import OSWorldACI

        self.platform = normalize_platform(args.os_type)
        self.screen_width = args.screen_width
        self.screen_height = args.screen_height

        main_engine_params = agent_s_engine_params(
            args,
            role="main",
            extra_body=extra_body,
        )
        grounding_engine_params = agent_s_engine_params(
            args,
            role="grounding",
            extra_body=extra_body,
        )
        code_engine_params = (
            agent_s_engine_params(args, role="code", extra_body=extra_body)
            if args.agent_s_enable_code_agent
            else None
        )
        aci_env = env if args.agent_s_enable_code_agent else None

        grounding_agent = OSWorldACI(
            env=aci_env,
            platform=self.platform,
            engine_params_for_generation=main_engine_params,
            engine_params_for_grounding=grounding_engine_params,
            width=self.screen_width,
            height=self.screen_height,
            code_agent_budget=args.agent_s_code_agent_budget,
            code_agent_engine_params=code_engine_params,
        )
        self.agent = AgentS3(
            main_engine_params,
            grounding_agent,
            platform=self.platform,
            max_trajectory_length=args.agent_s_max_trajectory_length,
            enable_reflection=args.agent_s_enable_reflection,
        )

    def predict(self, *, instruction: str, obs: dict[str, Any]) -> AgentStepPrediction:
        screenshot = screenshot_to_png_bytes(obs.get("screenshot"))
        agent_obs = dict(obs)
        agent_obs["screenshot"] = screenshot

        start = now()
        info, actions = self.agent.predict(instruction=instruction, observation=agent_obs)
        end = now()
        normalized_actions = normalize_agent_s_actions(actions)
        if not normalized_actions:
            normalized_actions = ["FAIL"]
        return AgentStepPrediction(
            info=dict(info or {}),
            actions=normalized_actions,
            latency_ms=latency_ms(start, end),
        )


def ensure_osworld_on_path(osworld_root: str) -> None:
    if osworld_root:
        path = Path(osworld_root).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"OSWorld source root not found: {path}. Set --osworld-root."
            )
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def maybe_load_dotenv(osworld_root: str) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    for path in (Path.cwd() / ".env", Path(osworld_root) / ".env" if osworld_root else None):
        if path and path.exists():
            load_dotenv(path)


def import_desktop_env(osworld_root: str) -> Any:
    ensure_osworld_on_path(osworld_root)
    try:
        from desktop_env.desktop_env import DesktopEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import OSWorld desktop_env. Clone OSWorld and pass "
            "--osworld-root, or add the checkout to PYTHONPATH. "
            f"Underlying missing module: {exc.name or exc}."
        ) from exc
    return DesktopEnv


def resolve_aws_snapshot_name(args: argparse.Namespace) -> str | None:
    if args.snapshot_name:
        return args.snapshot_name
    if args.provider_name != "aws":
        return None
    try:
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP
    except Exception:
        return None
    screen_size = (args.screen_width, args.screen_height)
    region_map = IMAGE_ID_MAP.get(args.region, {})
    return region_map.get(screen_size) or region_map.get((1920, 1080))


def create_desktop_env(args: argparse.Namespace) -> Any:
    DesktopEnv = import_desktop_env(args.osworld_root)
    snapshot_name = resolve_aws_snapshot_name(args)
    kwargs = {
        "path_to_vm": args.path_to_vm or None,
        "action_space": args.action_space,
        "provider_name": args.provider_name,
        "region": args.region,
        "screen_size": (args.screen_width, args.screen_height),
        "headless": args.headless,
        "os_type": args.os_type,
        "require_a11y_tree": args.observation_type
        in {"a11y_tree", "screenshot_a11y_tree", "som"},
        "enable_proxy": args.enable_proxy,
        "client_password": args.client_password,
    }
    if snapshot_name:
        kwargs["snapshot_name"] = snapshot_name
    return DesktopEnv(**kwargs)


def task_config_path(args: argparse.Namespace, task: TaskSpec) -> Path:
    return (
        Path(args.test_config_base_dir)
        / "examples"
        / task.domain
        / f"{task.example_id}.json"
    )


def load_task_config(args: argparse.Namespace, task: TaskSpec) -> dict[str, Any]:
    path = task_config_path(args, task)
    if not path.exists():
        raise FileNotFoundError(f"OSWorld task config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def maybe_start_recording(env: Any) -> None:
    controller = getattr(env, "controller", None)
    if controller is None or not hasattr(controller, "start_recording"):
        return
    try:
        controller.start_recording()
    except Exception:
        pass


def maybe_end_recording(env: Any, path: Path) -> None:
    controller = getattr(env, "controller", None)
    if controller is None or not hasattr(controller, "end_recording"):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        controller.end_recording(str(path))
    except Exception:
        pass


def reset_env(env: Any, task_config: dict[str, Any], *, reset_sleep: float) -> dict[str, Any]:
    output = env.reset(task_config=task_config)
    if reset_sleep > 0:
        time.sleep(reset_sleep)
    if isinstance(output, dict) and "screenshot" in output:
        return output
    return env._get_obs()


def step_env(
    env: Any,
    action: str,
    *,
    sleep_after_execution: float,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    output = env.step(action, sleep_after_execution)
    if isinstance(output, tuple):
        if len(output) == 4:
            obs, reward, done, info = output
            return obs, float(reward or 0.0), bool(done), dict(info or {})
        if len(output) == 5:
            obs, reward, terminated, truncated, info = output
            return obs, float(reward or 0.0), bool(terminated or truncated), dict(info or {})
    obs = env._get_obs()
    return obs, 0.0, action in {"DONE", "FAIL"}, {}


def evaluate_env(env: Any) -> float:
    result = env.evaluate()
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    try:
        return float(result)
    except Exception:
        return 0.0


def run_one_task(
    *,
    env: Any,
    args: argparse.Namespace,
    task: TaskSpec,
) -> EpisodeResult:
    task_id = task.instance_id
    task_output_dir = Path(args.output_dir) / task.domain / task.example_id
    trace_path = Path(args.output_dir) / "traces" / f"{task.domain}__{task.example_id}.jsonl"
    traj_path = task_output_dir / "traj.jsonl"
    result_path = task_output_dir / "result.json"
    recording_path = task_output_dir / "recording.mp4"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()
    if traj_path.exists():
        traj_path.unlink()

    program_id = ""
    score = 0.0
    step_count = 0
    error: str | None = None

    llm_kwargs: dict[str, Any] = {}
    extra_body = safe_json_object(args.extra_body, option_name="--extra-body")
    if extra_body:
        llm_kwargs["extra_body"] = extra_body

    with osworld_instance(
        llm_kwargs,
        instance_id=task_id,
        base_url=args.base_url,
        scaffold=args.scaffold,
    ) as (program, patched_llm_kwargs):
        program_id = program.program_id
        try:
            agent = AgentSOSWorldAgent(
                env=env,
                args=args,
                extra_body=patched_llm_kwargs.get("extra_body"),
            )
            task_config = load_task_config(args, task)
            instruction = str(task_config.get("instruction", ""))
            if not args.no_recording:
                maybe_start_recording(env)
            reset_start = now()
            obs = reset_env(env, task_config, reset_sleep=args.reset_sleep)
            reset_end = now()
            obs_sizes = observation_size(obs)
            write_jsonl(
                trace_path,
                trace_event(
                    task_id=task_id,
                    benchmark=args.benchmark,
                    framework=args.framework,
                    model=args.model,
                    step_id=0,
                    event_type="env_observation",
                    start=reset_start,
                    end=reset_end,
                    program_id=program_id,
                    observation_size_bytes=obs_sizes["total"],
                    state_delta={
                        "phase": "reset",
                        "obs_sizes": obs_sizes,
                        "instruction": instruction,
                        "agent_s_version": PINNED_AGENT_S_VERSION,
                    },
                ),
            )

            done = False
            for step_id in range(1, args.max_steps + 1):
                predict_start = now()
                prediction = agent.predict(instruction=instruction, obs=obs)
                predict_end = now()
                write_jsonl(
                    trace_path,
                    trace_event(
                        task_id=task_id,
                        benchmark=args.benchmark,
                        framework=args.framework,
                        model=args.model,
                        step_id=step_id,
                        event_type="llm_call",
                        start=predict_start,
                        end=predict_end,
                        program_id=program_id,
                        state_delta={
                            "agent_s_info": prediction.info,
                            "actions": prediction.actions,
                            "agent_s_version": PINNED_AGENT_S_VERSION,
                        },
                    ),
                )
                write_jsonl(
                    traj_path,
                    {
                        "step": step_id,
                        "instruction": instruction,
                        "agent_s_info": prediction.info,
                        "actions": prediction.actions,
                        "latency_ms": prediction.latency_ms,
                    },
                )

                for action in prediction.actions:
                    action_start = now()
                    obs, _reward, action_done, info = step_env(
                        env,
                        action,
                        sleep_after_execution=args.sleep_after_execution,
                    )
                    action_end = now()
                    tool_name = tool_name_from_action(action)
                    write_jsonl(
                        trace_path,
                        trace_event(
                            task_id=task_id,
                            benchmark=args.benchmark,
                            framework=args.framework,
                            model=args.model,
                            step_id=step_id,
                            event_type="tool_call",
                            start=action_start,
                            end=action_end,
                            program_id=program_id,
                            tool_name=tool_name,
                            tool_args_size=len(action.encode("utf-8")),
                            success=True,
                            state_delta={"action": action, "info": info},
                        ),
                    )
                    obs_sizes = observation_size(obs)
                    write_jsonl(
                        trace_path,
                        trace_event(
                            task_id=task_id,
                            benchmark=args.benchmark,
                            framework=args.framework,
                            model=args.model,
                            step_id=step_id,
                            event_type="env_observation",
                            start=action_end,
                            end=now(),
                            program_id=program_id,
                            observation_size_bytes=obs_sizes["total"],
                            success=True,
                            state_delta={"obs_sizes": obs_sizes, "done": action_done},
                        ),
                    )
                    if action_done or action in {"DONE", "FAIL"}:
                        done = True
                        break

                step_count = step_id
                if done:
                    break

            if args.settle_sleep > 0:
                time.sleep(args.settle_sleep)
            score = evaluate_env(env)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            write_jsonl(
                trace_path,
                trace_event(
                    task_id=task_id,
                    benchmark=args.benchmark,
                    framework=args.framework,
                    model=args.model,
                    step_id=step_count,
                    event_type="env_observation",
                    start=now(),
                    end=now(),
                    program_id=program_id,
                    success=False,
                    error_type=type(exc).__name__,
                    state_delta={
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                ),
            )
        finally:
            if not args.no_recording:
                maybe_end_recording(env, recording_path)

    result = EpisodeResult(
        task_id=task_id,
        domain=task.domain,
        example_id=task.example_id,
        program_id=program_id,
        score=float(score),
        success=score > 0,
        steps=step_count,
        output_dir=str(task_output_dir),
        trace_path=str(trace_path),
        error=error,
    )
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result.__dict__, handle, indent=2)
    return result


def run_batch(args: argparse.Namespace, tasks: list[TaskSpec]) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    env = None
    try:
        env = create_desktop_env(args)
        for task in tasks:
            results.append(run_one_task(env=env, args=args, task=task))
    except Exception as exc:
        for task in tasks:
            task_output_dir = Path(args.output_dir) / task.domain / task.example_id
            task_output_dir.mkdir(parents=True, exist_ok=True)
            result = EpisodeResult(
                task_id=task.instance_id,
                domain=task.domain,
                example_id=task.example_id,
                program_id="",
                score=0.0,
                success=False,
                steps=0,
                output_dir=str(task_output_dir),
                trace_path="",
                error=f"{type(exc).__name__}: {exc}",
            )
            with (task_output_dir / "result.json").open("w", encoding="utf-8") as handle:
                json.dump(result.__dict__, handle, indent=2)
            results.append(result)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return results


def load_test_meta(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        str(domain): [str(example_id) for example_id in examples]
        for domain, examples in payload.items()
    }


def resolve_task_ids(args: argparse.Namespace) -> list[TaskSpec]:
    meta_path = Path(args.test_all_meta_path)
    meta: dict[str, list[str]] | None = None

    if args.task_ids:
        raw_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
        tasks: list[TaskSpec] = []
        for raw_id in raw_ids:
            if "/" in raw_id:
                domain, example_id = raw_id.split("/", 1)
                tasks.append(TaskSpec(domain=domain, example_id=example_id))
                continue
            if args.domain != "all":
                tasks.append(TaskSpec(domain=args.domain, example_id=raw_id))
                continue
            if meta is None:
                meta = load_test_meta(meta_path)
            matches = [
                TaskSpec(domain=domain, example_id=raw_id)
                for domain, examples in meta.items()
                if raw_id in examples
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Task id {raw_id!r} is ambiguous or missing; use domain/example_id."
                )
            tasks.extend(matches)
        return tasks

    meta = load_test_meta(meta_path)
    tasks = [
        TaskSpec(domain=domain, example_id=example_id)
        for domain, examples in meta.items()
        if args.domain == "all" or args.domain == domain
        for example_id in examples
    ]
    if args.task_start:
        tasks = tasks[args.task_start :]
    if args.num_tasks:
        tasks = tasks[: args.num_tasks]
    return tasks


def partition_tasks(tasks: list[TaskSpec], max_concurrency: int) -> list[list[TaskSpec]]:
    batches = [[] for _ in range(max_concurrency)]
    for index, task in enumerate(tasks):
        batches[index % max_concurrency].append(task)
    return [batch for batch in batches if batch]


def write_run_summary(output_dir: Path, results: list[EpisodeResult]) -> None:
    scores = [result.score for result in results]
    summary = {
        "num_tasks": len(results),
        "num_errors": sum(1 for result in results if result.error),
        "success_rate": mean(scores) if scores else 0.0,
        "mean_steps": mean([result.steps for result in results]) if results else 0.0,
        "agent_s_version": PINNED_AGENT_S_VERSION,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def warn_runtime_tools() -> None:
    if shutil.which("tesseract") is None:
        print(
            "WARNING: tesseract was not found on PATH. Agent-S OCR/text-grounding "
            "actions may fail until the system package is installed.",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osworld-root", default=os.environ.get("OSWORLD_SOURCE_ROOT", ""))
    parser.add_argument("--test-config-base-dir", default="evaluation_examples")
    parser.add_argument("--test-all-meta-path", default="evaluation_examples/test_nogdrive.json")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1, help="0 means all selected tasks")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--framework", default="")
    parser.add_argument("--scaffold", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--agent-kind",
        choices=["agent-s", "opencua"],
        default=DEFAULT_AGENT_KIND,
    )

    parser.add_argument("--provider-name", default="aws", choices=["aws", "virtualbox", "vmware", "docker", "azure"])
    parser.add_argument("--path-to-vm", default="")
    parser.add_argument("--snapshot-name", default="")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--os-type", default="Ubuntu")
    parser.add_argument("--client-password", default="")
    parser.add_argument("--password", default="osworld-public-evaluation")
    parser.add_argument("--enable-proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-recording", action="store_true")

    parser.add_argument("--action-space", default="pyautogui")
    parser.add_argument(
        "--observation-type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
    )
    parser.add_argument("--sleep-after-execution", type=float, default=5.0)
    parser.add_argument("--reset-sleep", type=float, default=60.0)
    parser.add_argument("--settle-sleep", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=100)

    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--extra-body", default="")

    parser.add_argument("--agent-s-engine-type", default="openai")
    parser.add_argument("--agent-s-main-engine-type", default="")
    parser.add_argument("--agent-s-main-model", default="")
    parser.add_argument("--agent-s-main-base-url", default="")
    parser.add_argument("--agent-s-main-api-key", default="")
    parser.add_argument("--agent-s-main-extra-body", default="")
    parser.add_argument("--agent-s-grounding-engine-type", default="")
    parser.add_argument("--agent-s-grounding-model", default="")
    parser.add_argument("--agent-s-grounding-base-url", default="")
    parser.add_argument("--agent-s-grounding-api-key", default="")
    parser.add_argument("--agent-s-grounding-extra-body", default="")
    parser.add_argument("--agent-s-grounding-width", type=int, default=1920)
    parser.add_argument("--agent-s-grounding-height", type=int, default=1080)
    parser.add_argument("--agent-s-code-engine-type", default="")
    parser.add_argument("--agent-s-code-model", default="")
    parser.add_argument("--agent-s-code-base-url", default="")
    parser.add_argument("--agent-s-code-api-key", default="")
    parser.add_argument("--agent-s-code-extra-body", default="")
    parser.add_argument("--agent-s-code-agent-budget", type=int, default=20)
    parser.add_argument("--agent-s-enable-code-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--agent-s-max-trajectory-length", type=int, default=8)
    parser.add_argument("--agent-s-enable-reflection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-s-strict-version", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--opencua-model", default="")
    parser.add_argument("--opencua-base-url", default="")
    parser.add_argument("--opencua-api-key", default="")
    parser.add_argument("--opencua-extra-body", default="")
    parser.add_argument("--opencua-max-tokens", type=int, default=0)
    parser.add_argument(
        "--opencua-history-type",
        choices=["action_history", "thought_history", "observation_history"],
        default="action_history",
    )
    parser.add_argument(
        "--opencua-coordinate-type",
        choices=["relative", "qwen25"],
        default="qwen25",
    )
    parser.add_argument("--opencua-cot-level", choices=["l1", "l2", "l3"], default="l2")
    parser.add_argument("--opencua-max-image-history-length", type=int, default=3)
    parser.add_argument(
        "--opencua-use-old-sys-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recommended for OpenCUA-7B and OpenCUA-32B; disable for OpenCUA-72B.",
    )
    parser.add_argument("--opencua-request-timeout", type=float, default=500.0)
    parser.add_argument("--opencua-max-retries", type=int, default=20)
    return parser.parse_args()


def configure_agent_defaults(args: argparse.Namespace) -> None:
    if not args.framework:
        args.framework = DEFAULT_AGENT_FRAMEWORKS[args.agent_kind]
    if not args.scaffold:
        args.scaffold = DEFAULT_AGENT_SCAFFOLDS[args.agent_kind]
    if not args.opencua_model:
        args.opencua_model = args.model
    if not args.opencua_base_url:
        args.opencua_base_url = args.base_url
    if not args.opencua_api_key:
        args.opencua_api_key = args.api_key


def main() -> int:
    args = parse_args()
    configure_agent_defaults(args)
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.task_start < 0:
        raise ValueError("--task-start must be >= 0")
    if args.num_tasks < 0:
        raise ValueError("--num-tasks must be >= 0")
    if args.agent_kind == "agent-s":
        if args.agent_s_max_trajectory_length < 1:
            raise ValueError("--agent-s-max-trajectory-length must be >= 1")
        if args.agent_s_grounding_width < 1 or args.agent_s_grounding_height < 1:
            raise ValueError("--agent-s-grounding-width/height must be positive")
    if args.agent_kind == "opencua":
        if args.action_space != "pyautogui":
            raise ValueError("OpenCUA currently supports only --action-space pyautogui")
        if args.observation_type != "screenshot":
            raise ValueError("OpenCUA currently supports only --observation-type screenshot")
        if args.opencua_max_image_history_length < 1:
            raise ValueError("--opencua-max-image-history-length must be >= 1")
        if args.opencua_max_retries < 1:
            raise ValueError("--opencua-max-retries must be >= 1")

    ensure_osworld_on_path(args.osworld_root)
    maybe_load_dotenv(args.osworld_root)
    if args.agent_kind == "agent-s":
        validate_agent_s_version(args.agent_s_strict_version)
    warn_runtime_tools()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = resolve_task_ids(args)
    if not tasks:
        raise ValueError("No OSWorld tasks selected.")

    batches = partition_tasks(tasks, min(args.max_concurrency, len(tasks)))
    results_by_task: dict[str, EpisodeResult] = {}

    if len(batches) == 1:
        for result in run_batch(args, batches[0]):
            results_by_task[result.task_id] = result
    else:
        with ProcessPoolExecutor(max_workers=len(batches)) as executor:
            futures = {executor.submit(run_batch, args, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    results = future.result()
                except Exception as exc:
                    results = [
                        EpisodeResult(
                            task_id=task.instance_id,
                            domain=task.domain,
                            example_id=task.example_id,
                            program_id="",
                            score=0.0,
                            success=False,
                            steps=0,
                            output_dir=str(output_dir / task.domain / task.example_id),
                            trace_path="",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        for task in batch
                    ]
                for result in results:
                    results_by_task[result.task_id] = result

    ordered_results = [results_by_task[task.instance_id] for task in tasks]
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump([result.__dict__ for result in ordered_results], handle, indent=2)
    write_run_summary(output_dir, ordered_results)
    print(json.dumps([result.__dict__ for result in ordered_results], indent=2))
    return 0 if all(result.error is None for result in ordered_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
