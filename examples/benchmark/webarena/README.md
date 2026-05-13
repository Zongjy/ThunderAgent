# BrowserGym + WebArena-Verified Benchmark

This runner connects BrowserGym WebArena-Verified tasks to ThunderAgent. It starts
vLLM, starts ThunderAgent, runs a small ReAct BrowserGym scaffold, and writes
both ThunderAgent profiles and unified event traces.

## Setup

```bash
bash examples/scripts/setup_benchmark_env.sh webarena
```

You also need the official WebArena self-hosted services configured for
`browsergym-webarena-verified`.

Copy the env template and edit the host:

```bash
cp examples/benchmark/webarena/webarena.env.example \
  examples/benchmark/webarena/webarena.env
```

The runner loads `examples/benchmark/webarena/webarena.env` automatically. You
can override the path with `WEBARENA_ENV_FILE=/path/to/webarena.env`.

## Quick Start

```bash
WEBARENA_NUM_TASKS=100 \
WEBARENA_MAX_STEPS=5 \
bash examples/benchmark/webarena/run_browsergym_webarena_ta_default.sh
```

For the most robust run, pass full env IDs in `WEBARENA_TASK_IDS`. If
`WEBARENA_TASK_IDS` is not set, the runner generates original task IDs from
`WEBARENA_TASK_START` and `WEBARENA_NUM_TASKS`; the scaffold then resolves each
ID against BrowserGym's WebArena-Verified registry.

To reuse already running vLLM and ThunderAgent:

```bash
START_VLLM=0 START_THUNDERAGENT=0 VLLM_PORT=4747 TA_PORT=9000 \
WEBARENA_TASK_IDS=browsergym/webarena_verified.<intent_template_id>.<task_id>.<revision> \
bash examples/benchmark/webarena/run_browsergym_webarena_ta_default.sh
```

Use the scheduler wrapper for `--router tr`:

```bash
START_VLLM=0 \
WEBARENA_TASK_IDS=browsergym/webarena_verified.<intent_template_id>.<task_id>.<revision> \
bash examples/benchmark/webarena/run_browsergym_webarena_ta_scheduler.sh
```

## Main Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `WEBARENA_TASK_IDS` | empty | Comma-separated WebArena-Verified task IDs. Full env IDs are safest. |
| `WEBARENA_TASK_START` | `0` | First original WebArena task ID used when `WEBARENA_TASK_IDS` is empty. |
| `WEBARENA_NUM_TASKS` | `1` | Number of original task IDs generated when `WEBARENA_TASK_IDS` is empty. |
| `WEBARENA_SITE_FILTER` | empty | Optional exact site filter, such as `shopping`, used to generate matching task IDs. |
| `WEBARENA_BENCHMARK_MODULE` | `browsergym.webarena_verified` | BrowserGym benchmark registration module. |
| `WEBARENA_ENV_PREFIX` | `browsergym/webarena_verified` | Prefix used to resolve env IDs such as `browsergym/webarena_verified.<intent_template_id>.<task_id>.<revision>`. |
| `WEBARENA_MAX_STEPS` | `30` | Max browser actions per task. |
| `WEBARENA_OBSERVATION_MODE` | `axtree` | `axtree`, `dom`, or `both` prompt observation. |
| `WEBARENA_OBSERVATION_MAX_CHARS` | `24000` | Per-observation prompt truncation. |
| `WEBARENA_COMPACT_ACTIONS` | `0` | Set to `1` to shorten action descriptions in prompts. |

## Output

```text
examples/benchmark/webarena/runs/<run_name>/
├── manifest.env
├── browsergym_outputs/
│   ├── results.json
│   └── traces/*.jsonl
├── thunderagent_profiles/step_profiles.csv
├── logs/
│   ├── browsergym_webarena.log
│   ├── thunderagent_health.jsonl
│   ├── thunderagent_metrics.jsonl
│   └── vllm_metrics.prom
└── _analysis/
    ├── webarena_analysis.json
    └── webarena_events.csv
```

Each trace follows the unified event-level schema in `.agents/task.md` with
`llm_call`, `tool_call`, and `env_observation` events. LLM calls include the
ThunderAgent `program_id` through OpenAI `extra_body`.
