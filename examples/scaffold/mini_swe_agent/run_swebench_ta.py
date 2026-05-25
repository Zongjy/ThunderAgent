#!/usr/bin/env python3
"""Run mini-swe-agent SWE-bench tasks through ThunderAgent."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.adapters.mini_swe_agent import mini_swe_agent_instance
from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger


DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}

DOCKER_IMAGE_SOURCE = os.environ.get("EVAL_DOCKER_IMAGE_SOURCE", "epoch").strip().lower()
EPOCH_DOCKER_IMAGE_PREFIX = os.environ.get("EPOCH_DOCKER_IMAGE_PREFIX", "ghcr.io/epoch-research")

_PREDS_LOCK = threading.Lock()
_ROLLOUTS_LOCK = threading.Lock()


class ThunderAgentProgressTrackingAgent(DefaultAgent):
    """DefaultAgent with batch progress and optional trajectory checkpoints."""

    def __init__(
        self,
        *args: Any,
        progress_manager: RunBatchProgressManager,
        task_id: str,
        checkpoint_path: Path | None = None,
        checkpoint_interval_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.task_id = task_id
        self._checkpoint_path = checkpoint_path
        self._checkpoint_interval_steps = max(0, int(checkpoint_interval_steps))
        self._last_checkpoint_step = 0
        self.step_timings: list[dict[str, Any]] = []
        self._active_step_timing: dict[str, Any] | None = None

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.task_id,
            f"Step {self.n_calls + 1:3d} (${self.cost:.2f})",
        )
        step_timing: dict[str, Any] = {
            "step": self.n_calls + 1,
            "start_ts": time.time(),
        }
        self._active_step_timing = step_timing
        try:
            return super().step()
        finally:
            end_ts = time.time()
            step_timing["end_ts"] = end_ts
            step_timing["total_s"] = max(0.0, end_ts - step_timing["start_ts"])
            self.step_timings.append(step_timing)
            self._active_step_timing = None
            self._checkpoint_if_needed()

    def query(self) -> dict:
        timing = self._active_step_timing
        start_ts = time.time()
        try:
            return super().query()
        finally:
            end_ts = time.time()
            if timing is not None:
                timing["query_start_ts"] = start_ts
                timing["query_end_ts"] = end_ts
                timing["query_s"] = max(0.0, end_ts - start_ts)

    def execute_actions(self, message: dict) -> list[dict]:
        timing = self._active_step_timing
        start_ts = time.time()
        try:
            return super().execute_actions(message)
        finally:
            end_ts = time.time()
            if timing is not None:
                timing["tool_start_ts"] = start_ts
                timing["tool_end_ts"] = end_ts
                timing["tool_s"] = max(0.0, end_ts - start_ts)
                timing["action_count"] = len(message.get("extra", {}).get("actions", []))

    def _checkpoint_if_needed(self) -> None:
        if self._checkpoint_path is None or self._checkpoint_interval_steps <= 0:
            return
        step_idx = int(self.n_calls)
        if step_idx - self._last_checkpoint_step < self._checkpoint_interval_steps:
            return
        try:
            self.save(
                self._checkpoint_path,
                {
                    "info": {
                        "exit_status": "running",
                        "submission": "",
                    },
                    "instance_id": self.task_id,
                },
            )
            self._last_checkpoint_step = step_idx
        except Exception as exc:
            logger.warning("Failed to save checkpoint for %s: %s", self.task_id, exc)


def get_epoch_swebench_image(instance_id: str) -> str:
    return (
        f"{EPOCH_DOCKER_IMAGE_PREFIX.rstrip('/')}/"
        f"swe-bench.eval.x86_64.{instance_id}:latest"
    ).lower()


def get_official_swebench_image(instance_id: str) -> str:
    docker_id = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest".lower()


def get_swebench_docker_image_name(instance: dict[str, Any]) -> str:
    instance_id = instance["instance_id"]
    if DOCKER_IMAGE_SOURCE == "epoch":
        return get_epoch_swebench_image(instance_id)

    if DOCKER_IMAGE_SOURCE in {"", "official", "swebench", "dockerhub", "docker.io"}:
        image_name = instance.get("image_name") or instance.get("docker_image")
        if image_name is not None:
            return str(image_name)
        return get_official_swebench_image(instance_id)

    if DOCKER_IMAGE_SOURCE in {"instance", "dataset"}:
        image_name = instance.get("image_name") or instance.get("docker_image")
        if image_name is None:
            raise ValueError(
                f"EVAL_DOCKER_IMAGE_SOURCE={DOCKER_IMAGE_SOURCE!r} requires image_name "
                f"or docker_image for instance {instance_id}"
            )
        return str(image_name)

    raise ValueError(
        "Unsupported EVAL_DOCKER_IMAGE_SOURCE="
        f"{DOCKER_IMAGE_SOURCE!r}; expected epoch, official, dockerhub, instance, or dataset"
    )


def get_swebench_environment(config: dict[str, Any], instance: dict[str, Any]) -> Environment:
    env_config = copy.deepcopy(config.get("environment", {}))
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in {"docker", "swerex_modal"}:
        env_config["image"] = image_name
    elif env_config["environment_class"] in {"singularity", "contree"}:
        env_config["image"] = "docker://" + image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str) -> None:
    with _PREDS_LOCK:
        output_data = load_json_object(output_path)
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str) -> None:
    if not output_path.exists():
        return
    with _PREDS_LOCK:
        output_data = load_json_object(output_path)
        output_data.pop(instance_id, None)
        output_path.write_text(json.dumps(output_data, indent=2))


def update_rollouts_file(
    output_path: Path,
    instance_id: str,
    rollout_idx: int,
    model_name: str,
    result: str,
    *,
    task_id: str,
    exit_status: str,
) -> None:
    with _ROLLOUTS_LOCK:
        output_data = load_json_object(output_path)
        entries = output_data.get(instance_id, [])
        if not isinstance(entries, list):
            entries = []
        new_entry = {
            "rollout_idx": int(rollout_idx),
            "task_id": task_id,
            "model_name_or_path": model_name,
            "model_patch": result,
            "exit_status": exit_status,
        }
        for idx, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("rollout_idx") == rollout_idx:
                entries[idx] = new_entry
                break
        else:
            entries.append(new_entry)
        entries.sort(key=lambda entry: int(entry.get("rollout_idx", 0)) if isinstance(entry, dict) else 0)
        output_data[instance_id] = entries
        output_path.write_text(json.dumps(output_data, indent=2))


def get_completed_rollouts(output_path: Path, instance_id: str) -> set[int]:
    entries = load_json_object(output_path).get(instance_id, [])
    completed: set[int] = set()
    if not isinstance(entries, list):
        return completed
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            completed.add(int(entry.get("rollout_idx")))
        except Exception:
            pass
    return completed


def cleanup_swebench_image(env: Any, image_name: str) -> None:
    docker_executable = getattr(getattr(env, "config", None), "executable", "docker")
    subprocess.run(
        [docker_executable, "image", "rm", "-f", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dangling = subprocess.run(
        [docker_executable, "images", "-q", "--filter", "dangling=true"],
        text=True,
        capture_output=True,
    )
    for image_id in [line.strip() for line in dangling.stdout.splitlines() if line.strip()]:
        subprocess.run(
            [docker_executable, "image", "rm", "-f", image_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def docker_diagnostics_from_error(exc: BaseException) -> dict[str, Any]:
    """Capture Docker's real error text for failed container startup."""
    if not isinstance(exc, subprocess.CalledProcessError):
        return {}
    cmd = getattr(exc, "cmd", None)
    if not isinstance(cmd, list) or len(cmd) < 2 or cmd[0] != "docker" or cmd[1] != "run":
        return {}

    diagnostics: dict[str, Any] = {
        "returncode": exc.returncode,
        "cmd": cmd,
        "cmd_text": shlex.join(str(part) for part in cmd),
    }
    for field in ("stdout", "stderr", "output"):
        value = getattr(exc, field, None)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value:
            diagnostics[field] = str(value)

    return diagnostics


def process_instance(
    instance: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    progress_manager: RunBatchProgressManager,
    task_number: int,
    rollout_idx: int,
    rollouts_per_instance: int,
    checkpoint_interval_steps: int,
    cleanup_images: bool,
) -> None:
    instance_id = instance["instance_id"]
    task_id = instance_id if rollouts_per_instance <= 1 else f"{instance_id}#r{rollout_idx:02d}"
    instance_dir = output_dir / instance_id
    rollout_dir = instance_dir if rollouts_per_instance <= 1 else instance_dir / "rollouts" / f"r{rollout_idx:02d}"
    traj_path = rollout_dir / f"{instance_id}.traj.json"

    if rollouts_per_instance <= 1:
        remove_from_preds_file(output_dir / "preds.json", instance_id)
        traj_path.unlink(missing_ok=True)
    else:
        traj_path.unlink(missing_ok=True)

    model_config = copy.deepcopy(config.get("model", {}))
    model_config["step_limit"] = config.get("agent", {}).get("step_limit", 0)

    agent: ThunderAgentProgressTrackingAgent | None = None
    env: Environment | None = None
    model: Any | None = None
    exit_status = "unknown"
    result = ""
    extra_info: dict[str, Any] | None = None
    env_prepare_timing: dict[str, Any] | None = None

    with mini_swe_agent_instance(model_config, instance_id=task_id) as (program, patched_model_config):
        progress_manager.on_instance_start(task_id)
        progress_manager.update_instance_status(task_id, "Pulling/starting docker")

        try:
            model = get_model(config=patched_model_config)
            env_start_ts = time.time()
            try:
                env = get_swebench_environment(config, instance)
            finally:
                env_end_ts = time.time()
                env_prepare_timing = {
                    "start_ts": env_start_ts,
                    "end_ts": env_end_ts,
                    "total_s": max(0.0, env_end_ts - env_start_ts),
                    "environment_class": config.get("environment", {}).get("environment_class", "docker"),
                    "image": get_swebench_docker_image_name(instance),
                }

            agent = ThunderAgentProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                task_id=task_id,
                checkpoint_path=traj_path,
                checkpoint_interval_steps=checkpoint_interval_steps,
                **config.get("agent", {}),
            )
            info = agent.run(instance["problem_statement"])
            exit_status = info.get("exit_status", "")
            result = info.get("submission", "")
        except Exception as exc:
            logger.error("Error processing instance %s: %s", instance_id, exc, exc_info=True)
            exit_status, result = type(exc).__name__, ""
            extra_info = {
                "traceback": traceback.format_exc(),
                "exception_str": str(exc),
            }
            docker_diagnostics = docker_diagnostics_from_error(exc)
            if docker_diagnostics:
                extra_info["docker_diagnostics"] = docker_diagnostics
                stderr = docker_diagnostics.get("stderr")
                if stderr:
                    logger.error("Docker startup stderr for %s: %s", instance_id, stderr.strip())
        finally:
            rollout_dir.mkdir(parents=True, exist_ok=True)
            timing_payload = {
                "instance_id": instance_id,
                "task_id": task_id,
                "rollout_idx": rollout_idx,
                "rollouts_per_instance": rollouts_per_instance,
                "program_id": program.program_id,
                "task_number": task_number,
                "env_prepare": env_prepare_timing,
                "steps": getattr(agent, "step_timings", []) if agent is not None else [],
            }
            (rollout_dir / f"{instance_id}.timings.json").write_text(json.dumps(timing_payload, indent=2))
            if agent is not None:
                agent.save(
                    traj_path,
                    {
                        "info": {
                            "exit_status": exit_status,
                            "submission": result,
                            "program_id": program.program_id,
                            **(extra_info or {}),
                        },
                        "instance_id": instance_id,
                        "task_id": task_id,
                    },
                )
            else:
                traj_path.write_text(
                    json.dumps(
                        {
                            "info": {
                                "exit_status": exit_status,
                                "submission": result,
                                "program_id": program.program_id,
                                **(extra_info or {}),
                            },
                            "instance_id": instance_id,
                            "task_id": task_id,
                            "messages": [],
                            "trajectory_format": "mini-swe-agent-1.1",
                        },
                        indent=2,
                    )
                )
            logger.info("Saved trajectory to '%s'", traj_path)
            model_name = getattr(getattr(model, "config", None), "model_name", model_config.get("model_name", "unknown"))
            if rollouts_per_instance <= 1:
                update_preds_file(output_dir / "preds.json", instance_id, model_name, result)
            else:
                update_rollouts_file(
                    output_dir / "preds_rollouts.json",
                    instance_id,
                    rollout_idx,
                    model_name,
                    result,
                    task_id=task_id,
                    exit_status=exit_status,
                )
            if env and hasattr(env, "cleanup"):
                try:
                    env.cleanup()
                except Exception as cleanup_error:
                    logger.warning("Error during environment cleanup for %s: %s", instance_id, cleanup_error)
            if cleanup_images:
                try:
                    cleanup_swebench_image(env, get_swebench_docker_image_name(instance))
                except Exception as cleanup_error:
                    logger.warning("Error removing image for %s: %s", instance_id, cleanup_error)
            progress_manager.on_instance_end(task_id, exit_status)


def filter_instances(
    instances: list[dict[str, Any]],
    *,
    filter_spec: str,
    slice_spec: str,
    shuffle: bool,
) -> list[dict[str, Any]]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda item: item["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    if filter_spec:
        instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if len(instances) != before_filter:
        logger.info("Instance filter: %s -> %s instances", before_filter, len(instances))
    if slice_spec:
        values = [int(value) if value else None for value in slice_spec.split(":")]
        instances = instances[slice(*values)]
        logger.info("Instance slice: %s -> %s instances", before_filter, len(instances))
    return instances


def build_tasks(
    instances: list[dict[str, Any]],
    *,
    rollouts_per_instance: int,
    redo_existing: bool,
    rollouts_path: Path,
) -> list[tuple[dict[str, Any], int]]:
    if rollouts_per_instance <= 1:
        return [(instance, 1) for instance in instances]

    tasks: list[tuple[dict[str, Any], int]] = []
    for rollout_idx in range(1, max(1, rollouts_per_instance) + 1):
        for instance in instances:
            completed = set() if redo_existing else get_completed_rollouts(rollouts_path, instance["instance_id"])
            if redo_existing or rollout_idx not in completed:
                tasks.append((instance, rollout_idx))
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="lite", help="SWE-bench subset or dataset path")
    parser.add_argument("--split", default="dev", help="Dataset split")
    parser.add_argument("--slice", dest="slice_spec", default="", help="Slice, e.g. 0:5")
    parser.add_argument("--filter", dest="filter_spec", default="", help="Regex over instance_id")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle instances with a fixed seed")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Parallel worker count")
    parser.add_argument("--model", "-m", default=None, help="Override model name")
    parser.add_argument("--model-class", default=None, help="Override mini-swe-agent model class")
    parser.add_argument("--base-url", default=None, help="Override model base_url, normally ThunderAgent /v1")
    parser.add_argument("--config", required=True, help="mini-swe-agent YAML config")
    parser.add_argument("--environment-class", default=None, help="docker, singularity, ...")
    parser.add_argument("--cleanup-images", action="store_true", help="Remove SWE-bench image after each task")
    parser.add_argument("--rollouts-per-instance", type=int, default=1, help="Repeat each instance up to N times")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=0, help="Checkpoint trajectory every N steps")
    parser.add_argument("--redo-existing", action="store_true", help="Redo tasks already in preds.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "minisweagent.log")
    logger.info("Results will be saved to %s", output_path)
    logger.info(
        "SWE-bench docker image source: %s; Epoch prefix: %s",
        DOCKER_IMAGE_SOURCE or "official",
        EPOCH_DOCKER_IMAGE_PREFIX,
    )

    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    logger.info("Loading dataset %s, split %s...", dataset_path, args.split)
    instances = list(load_dataset(dataset_path, split=args.split))
    instances = filter_instances(
        instances,
        filter_spec=args.filter_spec,
        slice_spec=args.slice_spec,
        shuffle=args.shuffle,
    )

    single_preds_path = output_path / "preds.json"
    rollouts_path = output_path / "preds_rollouts.json"
    if not args.redo_existing and args.rollouts_per_instance <= 1 and single_preds_path.exists():
        existing = set(load_json_object(single_preds_path))
        logger.info("Skipping %s existing instances", len(existing))
        instances = [instance for instance in instances if instance["instance_id"] not in existing]

    config_path = get_config_path(Path(args.config))
    logger.info("Loading agent config from %s", config_path)
    config = yaml.safe_load(config_path.read_text()) or {}
    if args.environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = args.environment_class
    if args.model is not None:
        config.setdefault("model", {})["model_name"] = args.model
    if args.model_class is not None:
        config.setdefault("model", {})["model_class"] = args.model_class
    if args.base_url is not None:
        config.setdefault("model", {})["base_url"] = args.base_url

    tasks = build_tasks(
        instances,
        rollouts_per_instance=args.rollouts_per_instance,
        redo_existing=args.redo_existing,
        rollouts_path=rollouts_path,
    )
    logger.info(
        "Running %s tasks (%s instances, up to %s rollouts/instance)...",
        len(tasks),
        len(instances),
        args.rollouts_per_instance,
    )

    progress_manager = RunBatchProgressManager(len(tasks), output_path / f"exit_statuses_{time.time()}.yaml")

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures: dict[concurrent.futures.Future, str] = {}
            for task_number, (instance, rollout_idx) in enumerate(tasks, start=1):
                task_id = instance["instance_id"] if args.rollouts_per_instance <= 1 else f"{instance['instance_id']}#r{rollout_idx:02d}"
                futures[
                    executor.submit(
                        process_instance,
                        instance,
                        output_path,
                        config,
                        progress_manager,
                        task_number,
                        rollout_idx,
                        args.rollouts_per_instance,
                        args.checkpoint_interval_steps,
                        args.cleanup_images,
                    )
                ] = task_id

            try:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except concurrent.futures.CancelledError:
                        pass
                    except Exception as exc:
                        task_id = futures[future]
                        logger.error("Error in future for %s: %s", task_id, exc, exc_info=True)
                        progress_manager.on_uncaught_exception(task_id, exc)
            except KeyboardInterrupt:
                logger.info("Cancelling pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
