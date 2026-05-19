#!/usr/bin/env python3
"""Run WebArena through AgentLab with ThunderAgent as the LLM endpoint."""

from __future__ import annotations

import argparse
import copy
import importlib.resources
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.scaffold.agentlab_webarena.agent import (  # noqa: E402
    ThunderAgentGenericAgentArgs,
    ThunderAgentModelArgs,
)

logger = logging.getLogger("agentlab_webarena")

WA_TO_WEBARENA_ENV = {
    "WA_SHOPPING": "SHOPPING",
    "WA_SHOPPING_ADMIN": "SHOPPING_ADMIN",
    "WA_REDDIT": "REDDIT",
    "WA_GITLAB": "GITLAB",
    "WA_WIKIPEDIA": "WIKIPEDIA",
    "WA_MAP": "MAP",
    "WA_HOMEPAGE": "HOMEPAGE",
}


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_env_file(path: Path | None) -> None:
    if not path or not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = os.path.expandvars(value)


def prepare_webarena_env(site_filter: str | None) -> None:
    if not os.environ.get("WA_HOMEPAGE") and os.environ.get("WA_SHOPPING"):
        os.environ["WA_HOMEPAGE"] = os.environ["WA_SHOPPING"]

    selected_sites = set(split_csv(site_filter))
    allow_placeholders = selected_sites == {"shopping"}
    for wa_key, webarena_key in WA_TO_WEBARENA_ENV.items():
        if not os.environ.get(wa_key):
            if allow_placeholders:
                os.environ[wa_key] = "todo"
            else:
                raise RuntimeError(f"Missing required WebArena environment variable: {wa_key}")
        os.environ[webarena_key] = os.environ[wa_key]


def load_task_dataset(benchmark_name: str) -> list[dict[str, Any]]:
    if benchmark_name.startswith("webarena_verified"):
        return json.loads(
            importlib.resources.files("webarena_verified")
            .joinpath("assets/dataset/webarena-verified.json")
            .read_text()
        )

    return json.loads(importlib.resources.files("webarena").joinpath("test.raw.json").read_text())


def normalize_task_name(token: str, benchmark_name: str) -> str:
    token = token.strip()
    if "/" in token:
        token = token.rsplit("/", 1)[1]
    if re.fullmatch(r"\d+", token):
        return f"{benchmark_name}.{token}"
    return token


def select_task_names(args: argparse.Namespace, benchmark_name: str) -> list[str]:
    explicit = split_csv(args.task_ids)
    if explicit:
        return [normalize_task_name(item, benchmark_name) for item in explicit]

    dataset = load_task_dataset(benchmark_name)
    sites = set(split_csv(args.site_filter))
    if sites:
        matched = [
            str(item["task_id"])
            for item in dataset
            if set(item.get("sites", [])) == sites
        ]
    else:
        matched = [str(item["task_id"]) for item in dataset]

    start = args.task_start
    end = None if args.num_tasks == 0 else start + args.num_tasks
    selected = matched[start:end]
    if args.num_tasks and len(selected) < args.num_tasks:
        raise RuntimeError(
            f"Requested {args.num_tasks} tasks from offset {start}, "
            f"but only {len(matched)} matching tasks exist."
        )
    if not selected:
        raise RuntimeError("No WebArena tasks selected.")
    return [f"{benchmark_name}.{task_id}" for task_id in selected]


def patch_premature_fuzzy_validation() -> None:
    from browsergym.webarena.task import GenericWebArenaTask

    if getattr(GenericWebArenaTask, "_thunderagent_agentlab_validate_patch", False):
        return

    original_validate = GenericWebArenaTask.validate

    def config_uses_fuzzy_match(config: dict[str, Any]) -> bool:
        reference_answers = (config.get("eval") or {}).get("reference_answers") or {}
        return isinstance(reference_answers, dict) and "fuzzy_match" in reference_answers

    def has_final_chat_answer(chat_messages: list[dict[str, Any]]) -> bool:
        return bool(chat_messages) and chat_messages[-1].get("role") in {
            "assistant",
            "infeasible",
        }

    def validate_without_premature_fuzzy(self, page, chat_messages):
        if config_uses_fuzzy_match(self.config) and not has_final_chat_answer(chat_messages):
            import urllib.parse

            authorized_locations = ["newtab", ""] + [
                urllib.parse.urlparse(url).netloc
                for url in [*self.webarena_instance.urls.values(), self.webarena_instance.home_url]
            ]
            for open_page in page.context.pages:
                page_location = urllib.parse.urlparse(open_page.url).netloc
                if page_location not in authorized_locations:
                    return 0, True, "", {"error": "Unauthorized url, terminating task"}
            return 0.0, False, "", {}

        return original_validate(self, page, chat_messages)

    GenericWebArenaTask.validate = validate_without_premature_fuzzy
    GenericWebArenaTask._thunderagent_agentlab_validate_patch = True


def patch_webarena_evaluators(provider: str, base_url: str, model: str, api_key: str) -> None:
    if provider == "stock":
        return

    from webarena.evaluation_harness import evaluators, helper_functions

    if provider == "none":
        def no_external_judge(*_args: Any, **_kwargs: Any) -> float:
            return 0.0

        helper_functions.llm_fuzzy_match = no_external_judge
        helper_functions.llm_ua_match = no_external_judge
        evaluators.llm_fuzzy_match = no_external_judge
        evaluators.llm_ua_match = no_external_judge
        return

    if provider != "openai-compatible":
        raise ValueError(f"Unsupported evaluator provider: {provider}")

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    def judge(messages: list[dict[str, str]]) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_completion_tokens=768,
        )
        return (response.choices[0].message.content or "").lower()

    def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
        message = (
            "Grade whether the student answer is semantically equivalent to the "
            "reference answer. Reply with correct, incorrect, or partially correct.\n"
            f"question: {question}\nreference answer: {reference}\nstudent answer: {pred}"
        )
        response = judge(
            [
                {"role": "system", "content": "You are a strict grading assistant."},
                {"role": "user", "content": message},
            ]
        )
        return 0.0 if "incorrect" in response or "partially" in response else 1.0

    def llm_ua_match(pred: str, reference: str, question: str) -> float:
        message = (
            "Decide whether the reported unachievable reason matches the actual reason. "
            "Reply with same or different.\n"
            f"task: {question}\nactual reason: {reference}\nreported reason: {pred}"
        )
        response = judge(
            [
                {"role": "system", "content": "You are a strict grading assistant."},
                {"role": "user", "content": message},
            ]
        )
        return 0.0 if "different" in response else 1.0

    helper_functions.llm_fuzzy_match = llm_fuzzy_match
    helper_functions.llm_ua_match = llm_ua_match
    evaluators.llm_fuzzy_match = llm_fuzzy_match
    evaluators.llm_ua_match = llm_ua_match


def patch_shopping_only_backend_prepare(site_filter: str | None, skip_massage: bool) -> None:
    if set(split_csv(site_filter)) != {"shopping"}:
        return

    import browsergym.experiments.benchmark.base as benchmark_base
    import browsergym.experiments.benchmark.utils as benchmark_utils

    original_prepare_backend = benchmark_utils.prepare_backend

    def prepare_backend(backend: str):
        if backend != "webarena":
            return original_prepare_backend(backend)

        import browsergym.webarena  # noqa: F401
        from browsergym.webarena.instance import WebArenaInstance

        default_instance = WebArenaInstance()
        default_instance.full_reset()
        if skip_massage:
            logging.info("Skipping WebArena warm-up massage for shopping-only run.")
            return None

        logging.info("Massaging the shopping-only WebArena backend.")
        benchmark_utils.massage_tasks(["webarena.574"])
        return None

    benchmark_utils.prepare_backend = prepare_backend
    benchmark_base.prepare_backend = prepare_backend


def build_benchmark(args: argparse.Namespace):
    import browsergym.webarena  # noqa: F401
    from bgym import DEFAULT_BENCHMARKS

    benchmark = DEFAULT_BENCHMARKS[args.benchmark]()
    selected_names = select_task_names(args, benchmark.name)
    available = {env_args.task_name: env_args for env_args in benchmark.env_args_list}
    missing = [name for name in selected_names if name not in available]
    if missing:
        raise RuntimeError(f"Tasks are not in AgentLab benchmark {benchmark.name}: {missing[:10]}")

    env_args_list = []
    for task_name in selected_names:
        env_args = copy.deepcopy(available[task_name])
        env_args.max_steps = args.max_steps
        env_args.headless = args.headless
        env_args.task_seed = args.seed
        env_args.record_video = args.record_video
        env_args_list.append(env_args)
    benchmark.env_args_list = env_args_list
    return benchmark, selected_names


def build_agent_args(args: argparse.Namespace) -> ThunderAgentGenericAgentArgs:
    from agentlab.agents.generic_agent import AGENT_4o_MINI_VISION

    base = copy.deepcopy(AGENT_4o_MINI_VISION)
    flags = base.flags
    flags.max_prompt_tokens = args.max_prompt_tokens
    model_args = ThunderAgentModelArgs(
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_total_tokens=args.max_total_tokens,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_output_tokens,
        temperature=args.temperature,
        vision_support=True,
        log_probs=False,
        max_retry=args.llm_max_retry,
        min_retry_wait_time=args.llm_retry_wait,
    )
    return ThunderAgentGenericAgentArgs(
        chat_model_args=model_args,
        flags=flags,
        max_retry=args.action_parse_retry,
    )


def format_elapsed(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def count_completed_experiments(study_dir: Path) -> tuple[int, int]:
    completed = 0
    errors = 0
    for summary_path in study_dir.glob("*/summary_info.json"):
        completed += 1
        try:
            payload = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("err_msg"):
            errors += 1
    return completed, errors


def render_progress(completed: int, total: int, errors: int, started_at: float) -> str:
    width = 28
    ratio = min(1.0, completed / total) if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100
    elapsed = format_elapsed(time.time() - started_at)
    return (
        f"[AgentLab progress] [{bar}] {completed}/{total} "
        f"({percent:5.1f}%) errors={errors} elapsed={elapsed}"
    )


def start_progress_monitor(
    study_dir: Path,
    *,
    total: int,
    interval_s: int,
) -> tuple[threading.Event, threading.Thread | None]:
    stop_event = threading.Event()
    if interval_s <= 0:
        return stop_event, None

    def monitor() -> None:
        started_at = time.time()
        last_completed = -1
        last_errors = -1
        while not stop_event.wait(interval_s):
            completed, errors = count_completed_experiments(study_dir)
            if completed != last_completed or errors != last_errors:
                print(
                    render_progress(completed, total, errors, started_at),
                    flush=True,
                )
                last_completed = completed
                last_errors = errors

        completed, errors = count_completed_experiments(study_dir)
        print(render_progress(completed, total, errors, started_at), flush=True)

    thread = threading.Thread(target=monitor, name="agentlab-progress", daemon=True)
    thread.start()
    return stop_event, thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=env_default("WEBARENA_ENV_FILE", ""))
    parser.add_argument("--benchmark", default=env_default("WEBARENA_BENCHMARK", "webarena"))
    parser.add_argument("--task-ids", default=env_default("WEBARENA_TASK_IDS", ""))
    parser.add_argument("--site-filter", default=env_default("WEBARENA_SITE_FILTER", "shopping"))
    parser.add_argument("--task-start", type=int, default=env_int("WEBARENA_TASK_START", 0))
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=env_int("WEBARENA_NUM_TASKS", 0),
        help="Number of selected tasks to run; 0 means all matching tasks.",
    )
    parser.add_argument("--seed", type=int, default=env_int("WEBARENA_SEED", 0))
    parser.add_argument("--max-steps", type=int, default=env_int("WEBARENA_MAX_STEPS", 100))
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=env_bool("WEBARENA_HEADLESS", True))
    parser.add_argument("--record-video", action="store_true", default=env_bool("WEBARENA_RECORD_VIDEO", False))

    parser.add_argument("--base-url", default=env_default("THUNDERAGENT_BASE_URL", "http://127.0.0.1:9000/v1"))
    parser.add_argument("--api-key", default=env_default("THUNDERAGENT_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=env_default("AGENT_LLM", env_default("SERVED_MODEL_NAME", "Qwen/Qwen3.5-9B")))
    parser.add_argument("--temperature", type=float, default=float(env_default("AGENT_TEMPERATURE", "0")))
    parser.add_argument("--max-total-tokens", type=int, default=env_int("AGENT_MAX_TOTAL_TOKENS", 128000))
    parser.add_argument("--max-input-tokens", type=int, default=env_int("AGENT_MAX_INPUT_TOKENS", 128000))
    parser.add_argument("--max-output-tokens", type=int, default=env_int("AGENT_MAX_OUTPUT_TOKENS", 2048))
    parser.add_argument("--max-prompt-tokens", type=int, default=env_int("AGENT_MAX_PROMPT_TOKENS", 40000))
    parser.add_argument("--llm-max-retry", type=int, default=env_int("AGENT_LLM_MAX_RETRY", 4))
    parser.add_argument("--llm-retry-wait", type=int, default=env_int("AGENT_LLM_RETRY_WAIT", 5))
    parser.add_argument("--action-parse-retry", type=int, default=env_int("AGENT_ACTION_PARSE_RETRY", 4))

    parser.add_argument("--output-dir", default=env_default("AGENTLAB_EXP_ROOT", str(REPO_ROOT / "examples/benchmark/webarena/runs/agentlab_outputs")))
    parser.add_argument("--study-suffix", default=env_default("AGENTLAB_STUDY_SUFFIX", "thunderagent-shopping"))
    parser.add_argument("--comment", default=env_default("AGENTLAB_STUDY_COMMENT", "ThunderAgent AgentLab WebArena run"))
    parser.add_argument("--n-jobs", type=int, default=env_int("WEBARENA_MAX_CONCURRENCY", 1))
    parser.add_argument("--parallel-backend", default=env_default("AGENTLAB_PARALLEL_BACKEND", ""))
    parser.add_argument("--n-relaunch", type=int, default=env_int("AGENTLAB_N_RELAUNCH", 1))
    parser.add_argument("--ignore-dependencies", action="store_true", default=env_bool("AGENTLAB_IGNORE_DEPENDENCIES", False))
    parser.add_argument("--relaunch-errors", action="store_true", default=env_bool("AGENTLAB_RELAUNCH_ERRORS", False))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=env_int("AGENTLAB_PROGRESS_INTERVAL", 30))
    parser.add_argument("--skip-backend-massage", action="store_true", default=env_bool("WEBARENA_SKIP_BACKEND_MASSAGE", True))
    parser.add_argument("--no-skip-backend-massage", dest="skip_backend_massage", action="store_false")

    parser.add_argument("--evaluator-provider", default=env_default("WEBARENA_EVALUATOR_PROVIDER", "none"), choices=["none", "stock", "openai-compatible"])
    parser.add_argument("--evaluator-base-url", default=env_default("WEBARENA_EVALUATOR_BASE_URL", ""))
    parser.add_argument("--evaluator-model", default=env_default("WEBARENA_EVALUATOR_MODEL", ""))
    parser.add_argument("--evaluator-api-key", default=env_default("WEBARENA_EVALUATOR_API_KEY", env_default("OPENAI_API_KEY", "EMPTY")))
    parser.add_argument("--patch-premature-fuzzy-validation", action="store_true", default=env_bool("WEBARENA_PATCH_PREMATURE_FUZZY", True))
    parser.add_argument("--no-patch-premature-fuzzy-validation", dest="patch_premature_fuzzy_validation", action="store_false")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()

    env_file = Path(args.env_file) if args.env_file else None
    load_env_file(env_file)
    prepare_webarena_env(args.site_filter)

    os.environ.setdefault("AGENTLAB_EXP_ROOT", str(Path(args.output_dir).resolve()))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluator_provider == "openai-compatible":
        if not args.evaluator_base_url or not args.evaluator_model:
            raise RuntimeError(
                "--evaluator-provider=openai-compatible requires --evaluator-base-url and --evaluator-model"
            )
        patch_webarena_evaluators(
            args.evaluator_provider,
            args.evaluator_base_url,
            args.evaluator_model,
            args.evaluator_api_key,
        )
    else:
        patch_webarena_evaluators(args.evaluator_provider, "", "", "")

    if args.patch_premature_fuzzy_validation:
        patch_premature_fuzzy_validation()
    patch_shopping_only_backend_prepare(args.site_filter, args.skip_backend_massage)

    benchmark, selected_names = build_benchmark(args)
    agent_args = build_agent_args(args)
    parallel_backend = args.parallel_backend or ("ray" if args.n_jobs > 1 else "sequential")

    logger.info("Selected %d tasks: %s", len(selected_names), ",".join(selected_names[:10]))
    if len(selected_names) > 10:
        logger.info("... plus %d more tasks", len(selected_names) - 10)

    from agentlab.experiments.study import make_study

    study = make_study(
        benchmark=benchmark,
        agent_args=agent_args,
        suffix=args.study_suffix,
        comment=args.comment,
        ignore_dependencies=args.ignore_dependencies,
    )
    study.make_dir(exp_root=output_dir)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "benchmark": benchmark.name,
                    "study_name": study.name,
                    "experiment_count": len(study.exp_args_list),
                    "selected_task_count": len(selected_names),
                    "selected_tasks": selected_names,
                    "study_dir": str(study.dir),
                    "output_dir": str(output_dir),
                    "parallel_backend": parallel_backend,
                    "n_jobs": args.n_jobs,
                },
                indent=2,
            )
        )
        return 0

    stop_progress, progress_thread = start_progress_monitor(
        Path(study.dir),
        total=len(study.exp_args_list),
        interval_s=args.progress_interval,
    )
    try:
        study.run(
            n_jobs=args.n_jobs,
            parallel_backend=parallel_backend,
            n_relaunch=args.n_relaunch,
            relaunch_errors=args.relaunch_errors,
            exp_root=output_dir,
        )
    finally:
        stop_progress.set()
        if progress_thread is not None:
            progress_thread.join(timeout=5)
    print(f"AGENTLAB_STUDY_DIR={study.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
