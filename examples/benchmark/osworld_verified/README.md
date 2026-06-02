# OSWorld-Verified Benchmark

This benchmark runner is split into a long-lived vLLM launcher and a separate
ThunderAgent + OSWorld launcher. The model served by default is Qwen3.5-27B,
while the GUI-control scaffold remains OSWorld's Qwen3VL-style computer-use
agent.

Clone OSWorld first and point `OSWORLD_SOURCE_ROOT` at it:

```bash
git clone https://github.com/xlang-ai/OSWorld.git ~/OSWorld
OSWORLD_SOURCE_ROOT=~/OSWorld \
bash examples/scripts/setup_benchmark_env.sh osworld
```

Start vLLM first:

```bash
bash examples/benchmark/osworld_verified/vllm_serve_qwen35_27b.sh
```

Then run a one-task smoke test through the default router:

```bash
OSWORLD_SOURCE_ROOT=~/OSWorld \
OSWORLD_NUM_TASKS=1 \
OSWORLD_MAX_STEPS=100 \
bash examples/benchmark/osworld_verified/router_default.sh
```

Use the capacity-scheduling router with:

```bash
bash examples/benchmark/osworld_verified/router_tr.sh
```

Important defaults:

- `MODEL=Qwen/Qwen3.5-27B`
- `SERVED_MODEL_NAME=qwen3.5-27B`
- `AGENT_LLM=${SERVED_MODEL_NAME}`
- `VLLM_BASE_URL=http://127.0.0.1:${VLLM_PORT}`
- `QWEN3VL_BASE_URL=http://127.0.0.1:${TA_PORT}/v1`
- `QWEN3VL_MODEL=${AGENT_LLM}`
- `QWEN3VL_COORDINATE_TYPE=relative`
- `QWEN3VL_HISTORY_N=4`
- `QWEN3VL_ENABLE_THINKING=0`
- `QWEN3VL_USE_VLLM_TOOL_CALLS=1`
- `QWEN3VL_TOOL_CHOICE=named`
- `OSWORLD_MAX_OUTPUT_TOKENS=32768`
- `MAX_NUM_SEQS=32`
- `OSWORLD_MAX_STEPS=100`
- `OSWORLD_OBSERVATION_TYPE=screenshot`
- `OSWORLD_ACTION_SPACE=pyautogui`
- `OSWORLD_PROGRESS=1`
- `ENABLE_VLLM_TOOL_CALLING=1`
- `TOOL_CALL_PARSER=qwen3_coder`
- `VLLM_ENFORCE_STRICT_TOOL_CALLING=1`
- `THUNDERAGENT_KV_CAPACITY_TOKENS=380000`

Set `OSWORLD_TASK_IDS=domain/example_id` to run exact examples. Set
`OSWORLD_DOMAIN=<domain>` with `OSWORLD_TASK_START` and `OSWORLD_NUM_TASKS` to
select a slice from OSWorld's `test_nogdrive.json`.

## Native Runner

The ThunderAgent wrapper delegates the actual experiment loop to OSWorld:

```text
Manager.Queue of tasks
N multiprocessing.Process workers
each worker creates one DesktopEnv and one Qwen3VLAgent
lib_run_single.run_single_example(...)
env.reset -> agent.predict -> env.step -> env.evaluate
```

The runtime patch is on the model call: OSWorld's Qwen3VL agent uses the
OpenAI-compatible ThunderAgent endpoint, the wrapper sends the `computer_use`
tool schema to vLLM, reads `message.tool_calls`, and converts those calls back
to OSWorld pyautogui actions. It also attaches `extra_body.program_id` per task
so ThunderAgent can profile and schedule each desktop episode independently.

Native OSWorld results are written under
`<run>/osworld_outputs/summary/results.json`, with task artifacts under
`<run>/osworld_outputs/<action_space>/<observation_type>/<model>/<domain>/<id>/`.

The ThunderAgent wrapper also prints an overall progress line while the native
multi-process OSWorld runner is active. Worker processes append task start, step,
and end events to `<run>/osworld_outputs/_progress.jsonl`. Set
`OSWORLD_PROGRESS=0` to silence the progress line.
