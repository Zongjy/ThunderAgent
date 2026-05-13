# BrowserGym + WebArena-Verified Scaffold

This scaffold runs WebArena-Verified BrowserGym tasks with a small ReAct loop
and sends all LLM calls through ThunderAgent.

## Install

```bash
bash examples/scripts/setup_benchmark_env.sh webarena
```

WebArena-Verified also needs the official self-hosted sites and BrowserGym
configuration expected by `browsergym-webarena-verified`.

## Run One Task

```bash
python examples/scaffold/browsergym_webarena/run_browsergym_webarena.py \
  --base-url http://127.0.0.1:9000/v1 \
  --model Qwen/Qwen3.5-9B \
  --task-ids browsergym/webarena_verified.<intent_template_id>.<task_id>.<revision> \
  --output-dir /tmp/browsergym_webarena_smoke
```

Each BrowserGym episode receives a ThunderAgent `program_id` through
`extra_body`, and the scaffold writes unified event traces under
`<output-dir>/traces/*.jsonl`.

## Tool Set

The default `webarena` action subset is BrowserGym's WebArena-compatible high
level action set and is also used for WebArena-Verified:

`noop`, `scroll`, `keyboard_press`, `click`, `fill`, `hover`, `tab_focus`,
`new_tab`, `go_back`, `go_forward`, `goto`, `tab_close`, `select_option`,
`send_msg_to_user`, and `report_infeasible`.
