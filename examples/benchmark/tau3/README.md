# tau^3 Benchmark

End-to-end official tau2/τ³ benchmark with ThunderAgent + vLLM. This runner uses
the official `tau2.run_domain()` path and a thin ThunderAgent agent factory.

## Quick Start

```bash
bash run_tau3_ta_default.sh
```

By default this starts vLLM with `Qwen/Qwen3.5-9B`, starts ThunderAgent, and
runs the full `airline` `base` split. The script prefers local official data
from `/raid0/liyi/tau2-bench/data` when that checkout exists; otherwise it falls
back to `${REPO_ROOT}/.tau2_data` download flow.

Set `TAU3_NUM_TASKS` for a smoke test; leave it unset or set it to `0` for all
tasks in the selected split.

```bash
TAU3_DOMAIN=retail TAU3_NUM_TASKS=5 bash run_tau3_ta_default.sh
TAU3_DOMAIN=telecom TAU3_NUM_TASKS=5 bash run_tau3_ta_default.sh
TAU3_DOMAIN=banking_knowledge TAU3_RETRIEVAL_CONFIG=bm25 TAU3_NUM_TASKS=3 bash run_tau3_ta_default.sh
```

## Local vLLM

To reuse an already running local vLLM OpenAI-compatible server:

```bash
START_VLLM=0 VLLM_PORT=4747 TAU3_NUM_TASKS=5 bash run_tau3_ta_default.sh
```

The runner discovers the served model from `http://127.0.0.1:${VLLM_PORT}/v1/models`
and sets `AGENT_LLM=openai/<served-model>` and `USER_LLM=openai/<served-model>`.
If discovery is not available, set `SERVED_MODEL_NAME`, `AGENT_LLM`, and
`USER_LLM` explicitly.

## ThunderAgent Scheduling

Use the scheduler wrapper to run with `--router tr`:

```bash
START_VLLM=0 VLLM_PORT=4747 TAU3_MAX_CONCURRENCY=4 TAU3_NUM_TASKS=20 bash run_tau3_ta_scheduler.sh
```

Set `ROUTER_MODE=default` only when you want pure proxy behavior without
capacity scheduling.

## Analysis

```bash
python analyze_tau3_outputs.py --results runs/<name>/official/results.json
```
