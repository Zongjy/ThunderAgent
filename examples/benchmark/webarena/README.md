# AgentLab + WebArena Benchmark

This runner uses AgentLab as the WebArena harness. AgentLab creates the
WebArena study, runs AgentLab's browser environments, stores AgentLab traces, and
sends LLM calls through ThunderAgent with one `program_id` per task.

## Setup

```bash
bash examples/scripts/setup_benchmark_env.sh webarena
```

WebArena sites are still self-hosted. For the current shopping-only deployment,
edit `examples/benchmark/webarena/webarena.env` so `WA_SHOPPING` points at the
shopping service. Leave the other `WA_*` variables as `todo`; the launcher
defaults to `WEBARENA_SITE_FILTER=shopping`.

## Run All Shopping Tasks

Default router:

```bash
bash examples/benchmark/webarena/run_agentlab_webarena_default.sh
```

TR scheduler:

```bash
bash examples/benchmark/webarena/run_agentlab_webarena_tr.sh
```

The defaults select every WebArena task whose `sites == ["shopping"]`
(`WEBARENA_NUM_TASKS=0`). To run a smaller smoke test:

```bash
WEBARENA_TASK_IDS=21 WEBARENA_MAX_STEPS=5 \
bash examples/benchmark/webarena/run_agentlab_webarena_default.sh
```

To reuse already running vLLM and ThunderAgent:

```bash
START_VLLM=0 START_THUNDERAGENT=0 VLLM_PORT=4747 TA_PORT=9000 \
WEBARENA_TASK_IDS=21,22,23 \
bash examples/benchmark/webarena/run_agentlab_webarena_default.sh
```

## Main Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `WEBARENA_TASK_IDS` | empty | Comma-separated WebArena task IDs such as `21,22,23`. |
| `WEBARENA_SITE_FILTER` | `shopping` | Exact site set used when task IDs are empty. |
| `WEBARENA_NUM_TASKS` | `0` | Number of matching tasks; `0` means all matching tasks. |
| `WEBARENA_TASK_START` | `0` | Offset into the matching task list. |
| `WEBARENA_MAX_STEPS` | `100` | Max browser actions per task. |
| `WEBARENA_MAX_CONCURRENCY` | `1` | AgentLab job count. |
| `WEBARENA_SKIP_BACKEND_MASSAGE` | `1` | Skip AgentLab's cross-site warm-up tasks that touch non-shopping sites. |
| `AGENTLAB_PARALLEL_BACKEND` | auto | `sequential` for one job, `ray` for more than one job. |
| `AGENTLAB_PROGRESS_INTERVAL` | `30` | Seconds between progress updates; set `0` to disable. |
| `WEBARENA_EVALUATOR_PROVIDER` | `none` | `none`, `stock`, or `openai-compatible` fuzzy-answer evaluator. |

## Output

```text
examples/benchmark/webarena/runs/<run_name>/
├── manifest.env
├── agentlab_outputs/
│   └── <agentlab-study-dir>/
├── thunderagent_profiles/step_profiles.csv
└── logs/
    ├── agentlab_webarena.log
    ├── thunderagent_health.jsonl
    ├── thunderagent_metrics.jsonl
    └── vllm_metrics.prom
```

Use AgentLab's native analysis tooling on the printed study directory:

```python
from agentlab.analyze import inspect_results
df = inspect_results.load_result_df("/path/to/agentlab-study-dir")
print(df.head())
```
