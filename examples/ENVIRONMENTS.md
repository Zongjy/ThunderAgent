# Benchmark Environments

Use one uv environment per benchmark scaffold. This keeps OpenHands, tau2/tau3,
and BrowserGym/WebArena-Verified dependency constraints from fighting inside one shared
`.venv`.

## Recommended Layout

```text
/raid0/liyi/ThunderAgent/
├── .venv-openhands    # SWE-bench + OpenHands
├── .venv-tau3         # tau2/tau3
├── .venv-webarena     # BrowserGym + WebArena-Verified
└── .venv              # optional small ThunderAgent/dev env

~/vllm/.venv           # vLLM serving env, kept separate
```

Each benchmark env installs ThunderAgent editable, so the runner can start
`python -m ThunderAgent` from that env. The agent requests still go through
`http://127.0.0.1:${TA_PORT}/v1` and carry `extra_body.program_id`; they do not
need to share one Python environment.

## Build Environments

The setup helper defaults `UV_CACHE_DIR` to `/tmp/uv-cache-thunderagent` to avoid
read-only cache issues from older uv cache directories.

```bash
cd /raid0/liyi/ThunderAgent

bash examples/scripts/setup_benchmark_env.sh swebench
bash examples/scripts/setup_benchmark_env.sh tau3
bash examples/scripts/setup_benchmark_env.sh webarena
```

For WebArena-Verified, set `INSTALL_PLAYWRIGHT=0` if the Chromium browser is already
installed or you want to install it manually later.

```bash
INSTALL_PLAYWRIGHT=0 bash examples/scripts/setup_benchmark_env.sh webarena
```

## Runner Defaults

The benchmark launchers now default to these envs:

| Benchmark | Default env | Override |
| --- | --- | --- |
| SWE-bench/OpenHands | `.venv-openhands` | `OPENHANDS_ENV_DIR` or `PYTHON_BIN` |
| tau2/tau3 | `.venv-tau3` | `TAU3_ENV_DIR` or `PYTHON_BIN` |
| BrowserGym/WebArena-Verified | `.venv-webarena` | `WEBARENA_ENV_DIR` or `PYTHON_BIN` |

Examples:

```bash
SWEBENCH_LIMIT=1 SWEBENCH_WORKERS=1 \
bash examples/benchmark/swebench/run_openhands_ta_default.sh

TAU3_NUM_TASKS=3 \
bash examples/benchmark/tau3/run_tau3_ta_default.sh

WEBARENA_TASK_IDS=0 WEBARENA_MAX_STEPS=5 \
bash examples/benchmark/webarena/run_browsergym_webarena_ta_default.sh
```

You can always point a runner at a custom env:

```bash
PYTHON_BIN=/path/to/env/bin/python \
bash examples/benchmark/webarena/run_browsergym_webarena_ta_default.sh
```

## Dependency Manifests

| Scaffold | Dependency manifest |
| --- | --- |
| OpenHands | `examples/scaffold/openhands/pyproject.toml` + `uv.lock` |
| tau2/tau3 | `examples/scaffold/tau3/requirements.txt` |
| BrowserGym/WebArena-Verified | `examples/scaffold/browsergym_webarena/requirements.txt` |

Keep benchmark-specific dependencies in those scaffold directories rather than
adding them to the root ThunderAgent `pyproject.toml`.
