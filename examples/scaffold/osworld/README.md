# OSWorld Qwen3.5 Wrapper

This scaffold is a thin ThunderAgent wrapper around OSWorld's native
`scripts/python/run_multienv_qwen3vl.py`. OSWorld still owns the experiment
loop, multiprocessing queue, `DesktopEnv` lifecycle, trajectory files, videos,
and `env.evaluate()` calls.

The wrapper only selects an optional task subset and patches OSWorld's
`mm_agents/qwen3vl_agent.py` so its OpenAI-compatible calls go through
ThunderAgent with one `program_id` per OSWorld task.

## Setup

```bash
git clone https://github.com/xlang-ai/OSWorld.git ~/OSWorld
OSWORLD_SOURCE_ROOT=~/OSWorld \
bash examples/scripts/setup_benchmark_env.sh osworld
```

## Run One Task

Start or reuse a ThunderAgent OpenAI-compatible endpoint, then run:

```bash
python examples/scaffold/osworld/run_osworld_verified.py \
  --osworld-root ~/OSWorld \
  --test-config-base-dir ~/OSWorld/evaluation_examples \
  --test-all-meta-path ~/OSWorld/evaluation_examples/test_nogdrive.json \
  --base-url http://127.0.0.1:9000/v1 \
  --model qwen3.5-27B \
  --domain libreoffice \
  --num-tasks 1 \
  --max-steps 100 \
  --output-dir /tmp/osworld_verified_smoke
```

## Parallel Runs

Use `--max-concurrency N`; the wrapper maps this to OSWorld's native
`--num_envs N`. Each OSWorld worker process owns one `DesktopEnv`, one agent,
and consumes tasks from OSWorld's shared multiprocessing queue.

For local VM providers, use `N=1` unless you have provisioned independent VMs or
containers. For cloud providers, make sure quota and cleanup are ready before
raising concurrency.

## Progress

The wrapper prints an overall task progress line by default while OSWorld's
native multi-process runner is active. It records task start, step, and end
events in `<output-dir>/_progress.jsonl`; use `--no-progress` to disable the
terminal progress line.
