# OSWorld-Verified Benchmark

This benchmark runner starts vLLM, ThunderAgent, and the OSWorld Agent-S
scaffold. OSWorld still provides `DesktopEnv`, reset/evaluation logic, and
task metadata; Agent-S is now the only GUI agent loop.

Clone OSWorld first and point `OSWORLD_SOURCE_ROOT` at it:

```bash
git clone https://github.com/xlang-ai/OSWorld.git /raid0/liyi/OSWorld
OSWORLD_SOURCE_ROOT=/raid0/liyi/OSWorld \
bash examples/scripts/setup_benchmark_env.sh osworld
```

Run a one-task smoke test:

```bash
OSWORLD_SOURCE_ROOT=/raid0/liyi/OSWorld \
OSWORLD_NUM_TASKS=1 \
OSWORLD_MAX_STEPS=100 \
bash examples/benchmark/osworld_verified/run_osworld_verified_default.sh
```

Important defaults:

- `MODEL=ByteDance-Seed/UI-TARS-1.5-7B`
- `AGENT_LLM=${SERVED_MODEL_NAME}`
- `VLLM_EXTRA_ARGS=--trust-remote-code`
- `MAX_MODEL_LEN=32768`
- `REASONING_PARSER=` unset
- `OSWORLD_TEMPERATURE=0`
- `OSWORLD_TOP_P=0.9`
- `OSWORLD_MAX_OUTPUT_TOKENS=1000`
- `OSWORLD_MAX_STEPS=100`
- `OSWORLD_OBSERVATION_TYPE=screenshot`
- `OSWORLD_ACTION_SPACE=pyautogui`
- `AGENT_S_VERSION=0.3.2`
- `AGENT_S_ENGINE_TYPE=openai`
- `AGENT_S_BASE_URL=http://127.0.0.1:${TA_PORT}/v1`
- `AGENT_S_MAIN_MODEL=${AGENT_LLM}`
- `AGENT_S_GROUNDING_MODEL=${AGENT_LLM}`
- `AGENT_S_ENABLE_CODE_AGENT=0`
- `AGENT_S_MAX_TRAJECTORY_LENGTH=8`
- `AGENT_S_ENABLE_REFLECTION=1`
- `AGENT_S_STRICT_VERSION=1`
- `ENABLE_VLLM_LANGUAGE_MODEL_ONLY=0`
- `ENABLE_VLLM_TOOL_CALLING=0`

Set `OSWORLD_TASK_IDS=domain/example_id` to run exact examples. Set
`OSWORLD_DOMAIN=<domain>` with `OSWORLD_TASK_START` and `OSWORLD_NUM_TASKS` to
select a slice from OSWorld's `test_nogdrive.json`.

## Agent-S Pin

The setup helper installs `examples/scaffold/osworld/requirements.txt`, which
pins `gui-agents==0.3.2`. The launcher checks this before starting services
when `AGENT_S_STRICT_VERSION=1`, and the Python runner repeats the same check
inside worker processes.

Install `tesseract-ocr` on the host if Agent-S text grounding/OCR is used.
Without it, the runner will warn and Agent-S OCR actions may fail.

## Parallel Experiments

Set `OSWORLD_MAX_CONCURRENCY=N` to run up to `N` OSWorld tasks in parallel.
Each worker process owns a separate `DesktopEnv` and a separate ThunderAgent
`program_id`, while all workers share the same ThunderAgent and vLLM services.

Expected constraints:

- Local VirtualBox/VMware/Docker providers usually need one independent VM or
  container per worker; otherwise runs can collide on the same desktop state.
- AWS parallel runs need enough instance quota and cleanup discipline in the
  selected region.
- vLLM GPU memory and throughput can become the bottleneck because Agent-S may
  make multiple LLM calls per OSWorld step.
- ThunderAgent can schedule multiple programs, but all requests must preserve
  `extra_body.program_id`; this runner patches Agent-S engines to do that.

Start with `OSWORLD_MAX_CONCURRENCY=1` for smoke tests, then raise it after the
provider and model server are stable.
