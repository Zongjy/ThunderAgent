# AgentLab + WebArena Scaffold

This scaffold runs WebArena through AgentLab `make_study()` and routes every
LLM call to ThunderAgent's OpenAI-compatible endpoint. One AgentLab task maps to
one ThunderAgent `program_id`.

## Setup

```bash
bash examples/scripts/setup_benchmark_env.sh webarena
python -m playwright install chromium
```

WebArena sites still need to be deployed separately. For a shopping-only
deployment, set `WA_SHOPPING` and `WA_HOMEPAGE`; leave the other `WA_*` values
as `todo` and run with `WEBARENA_SITE_FILTER=shopping`.

## Smoke Check

```bash
WEBARENA_TASK_IDS=21 \
START_VLLM=0 START_THUNDERAGENT=0 \
bash examples/benchmark/webarena/run_agentlab_webarena_default.sh
```

For a dry run that only resolves AgentLab tasks:

```bash
python examples/scaffold/agentlab_webarena/run_agentlab_webarena.py \
  --env-file examples/benchmark/webarena/webarena.env \
  --site-filter shopping \
  --num-tasks 0 \
  --dry-run
```
