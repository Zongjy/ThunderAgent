# SWE-bench + OpenHands 实验步骤

## 目标

先跑 `SWE-bench Verified + OpenHands + ThunderAgent(default router) + vLLM`，收集：

- OpenHands 原始轨迹与结果
- ThunderAgent `step_profiles.csv`
- ThunderAgent `/health`、`/metrics` 采样
- vLLM `/metrics` 采样

## 环境准备

```bash
cd /raid0/liyi/ThunderAgent
bash examples/scripts/setup_benchmark_env.sh swebench
```

默认会分开使用两个环境：

- vLLM：`~/vllm/.venv/bin/vllm`
- ThunderAgent/OpenHands：`/raid0/liyi/ThunderAgent/.venv-openhands/bin/python`

如果路径不同，改下面参数表里的 `VLLM_*`、`OPENHANDS_ENV_DIR` 或 `PYTHON_BIN`。

## 快速运行

```bash
cd /raid0/liyi/ThunderAgent
bash examples/benchmark/swebench/run_openhands_ta_default.sh
```

默认配置：

- 模型：`Qwen/Qwen3.5-9B`
- OpenHands：`CodeActAgent + max_iter=100 + no hint/no plan/no ICL`
- vLLM：`127.0.0.1:4747`
- vLLM recipe：`qwen3_coder` tool parser + `qwen3` reasoning parser + text-only
- ThunderAgent：`127.0.0.1:9000`
- router：`default`
- SWE-bench Verified：100 个任务
- OpenHands workers：4
- max iterations：100

## 可修改参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen3.5-9B` | vLLM 加载的模型 |
| `SERVED_MODEL_NAME` | 同 `MODEL` | OpenAI API 中使用的模型名 |
| `VLLM_PORT` | `4747` | vLLM 端口 |
| `TA_PORT` | `9000` | ThunderAgent 端口 |
| `ROUTER_MODE` | `default` | `default` 或 `tr` |
| `SWEBENCH_DATASET` | `princeton-nlp/SWE-bench_Verified` | SWE-bench 数据集 |
| `SWEBENCH_SPLIT` | `test` | 数据 split |
| `SWEBENCH_LIMIT` | `100` | 任务数；`0` 表示全量 |
| `SWEBENCH_WORKERS` | `4` | OpenHands 并发 worker 数 |
| `SWEBENCH_MAX_ITERATIONS` | `100` | 每个任务最大 agent 轮数 |
| `SWEBENCH_MODE` | `swe` | OpenHands 评测模式：`swe`、`swt`、`swt-ci` |
| `OPENHANDS_AGENT_CLS` | `CodeActAgent` | OpenHands agent class |
| `OPENHANDS_AGENT_CONFIG` | `swe_lego_no_plan` | 写入并传给 `--agent-config` 的配置名 |
| `OPENHANDS_ENABLE_PLAN_MODE` | `false` | 是否启用 OpenHands plan mode |
| `OPENHANDS_NATIVE_TOOL_CALLING` | `true` | 是否使用 native tool calling；默认不走 mock function calling，也不插 ICL 示例 |
| `OPENHANDS_LLM_TIMEOUT` | `240` | OpenHands 单次 LLM 请求超时秒数 |
| `OPENHANDS_EVAL_NOTE` | `no-hint-no-plan-no-icl` | OpenHands 输出目录中的 eval note |
| `USE_HINT_TEXT` | `false` | 是否使用 SWE-bench hint text |
| `START_VLLM` | `1` | 是否由脚本启动 vLLM |
| `START_THUNDERAGENT` | `1` | 是否由脚本启动 ThunderAgent |
| `KEEP_SERVICES` | `0` | 跑完后是否保留服务 |
| `SAMPLE_METRICS` | `1` | 是否采样 `/health` 和 `/metrics` |
| `METRICS_INTERVAL_S` | `5` | 采样间隔秒数 |
| `HEALTH_TIMEOUT_S` | `1800` | 服务启动等待超时 |
| `OPENHANDS_ENV_DIR` | `/raid0/liyi/ThunderAgent/.venv-openhands` | SWE-bench/OpenHands 独立 uv 环境 |
| `THUNDERAGENT_ENV_DIR` | 同 `OPENHANDS_ENV_DIR` | 启动 ThunderAgent 的虚拟环境；默认与 benchmark env 相同 |
| `PYTHON_BIN` | `${THUNDERAGENT_ENV_DIR}/bin/python` | 启动 ThunderAgent 和 OpenHands 的 Python |
| `VLLM_ROOT` | `~/vllm` | vLLM 源码/环境目录 |
| `VLLM_ENV_DIR` | `~/vllm/.venv` | vLLM 虚拟环境 |
| `VLLM_WORKDIR` | `~/vllm` | 执行 `vllm serve` 时的工作目录 |
| `VLLM_BIN` | `~/vllm/.venv/bin/vllm` | vLLM 命令路径 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Hugging Face endpoint |
| `GPU_MEMORY_UTILIZATION` | `0.85` | vLLM GPU 显存占比 |
| `MAX_MODEL_LEN` | `262144` | vLLM 最大上下文长度 |
| `ENABLE_PREFIX_CACHING` | `1` | 是否开启 prefix cache |
| `ENABLE_VLLM_USAGE_FLAGS` | `1` | 是否开启 usage/token details |
| `ENABLE_VLLM_TOOL_CALLING` | `1` | 是否开启 vLLM tool calling |
| `ENABLE_VLLM_KV_METRICS` | `1` | 是否开启 vLLM KV/MFU metrics |
| `ENABLE_VLLM_LANGUAGE_MODEL_ONLY` | `1` | 是否传入 vLLM `--language-model-only` |
| `TOOL_CALL_PARSER` | `qwen3_coder` | vLLM tool-call parser |
| `REASONING_PARSER` | `qwen3` | vLLM reasoning parser |
| `VLLM_EXTRA_ARGS` | 空 | 追加给 `vllm serve` 的参数 |
| `EVAL_DOCKER_IMAGE_SOURCE` | `epoch` | OpenHands SWE-bench 镜像来源；默认使用 Epoch GHCR 镜像 |
| `EPOCH_DOCKER_IMAGE_PREFIX` | `ghcr.io/epoch-research` | Epoch SWE-bench 镜像前缀 |
| `RUN_NAME` | 自动时间戳 | 本次运行名 |
| `OUTPUT_ROOT` | `examples/benchmark/swebench/runs` | 输出根目录 |

## 例子

先用 1 个任务做 native tool-calling sanity check：

```bash
RUN_NAME=sanity_qwen35_9b SWEBENCH_LIMIT=1 SWEBENCH_WORKERS=1 \
bash examples/benchmark/swebench/run_openhands_ta_default.sh
```

如果要临时切回 SWE-Lego 并使用 mock function calling 但关闭 ICL：

```bash
RUN_NAME=sanity_swelego_mock_no_icl SWEBENCH_LIMIT=1 SWEBENCH_WORKERS=1 \
MODEL=SWE-Lego/SWE-Lego-Qwen3-8B \
OPENHANDS_NATIVE_TOOL_CALLING=false ENABLE_VLLM_TOOL_CALLING=0 \
SERVED_MODEL_NAME=local-openhands-lm/SWE-Lego-Qwen3-8B \
bash examples/benchmark/swebench/run_openhands_ta_default.sh
```

```bash
SWEBENCH_LIMIT=100 SWEBENCH_WORKERS=4 \
bash examples/benchmark/swebench/run_openhands_ta_default.sh
```

复用已经启动的 vLLM 和 ThunderAgent：

```bash
START_VLLM=0 START_THUNDERAGENT=0 VLLM_PORT=4747 TA_PORT=9000 \
bash examples/benchmark/swebench/run_openhands_ta_default.sh
```

跑完后保留服务：

```bash
KEEP_SERVICES=1 bash examples/benchmark/swebench/run_openhands_ta_default.sh
```

如果 vLLM 版本不支持 usage 相关参数：

```bash
ENABLE_VLLM_USAGE_FLAGS=0 bash examples/benchmark/swebench/run_openhands_ta_default.sh
```

换成 backup 模型：

```bash
MODEL=Qwen/Qwen3-8B MAX_MODEL_LEN=128K TOOL_CALL_PARSER=qwen3_coder REASONING_PARSER=qwen3 \
bash examples/benchmark/swebench/run_openhands_ta_default.sh
```

## 输出位置

每次运行会生成：

```text
examples/benchmark/swebench/runs/<run_name>/
├── manifest.env
├── openhands_config.toml
├── openhands_outputs/
├── thunderagent_profiles/
│   └── step_profiles.csv
└── logs/
    ├── vllm_*.log
    ├── thunderagent_*.log
    ├── openhands_swebench.log
    ├── thunderagent_health.jsonl
    ├── thunderagent_metrics.jsonl
    └── vllm_metrics.prom
```

## 分析结果

运行结束后执行：

```bash
python examples/benchmark/swebench/analyze_swebench_outputs.py \
  --root examples/benchmark/swebench/runs/<run_name> \
  --stdout-json
```

默认输出：

```text
examples/benchmark/swebench/runs/<run_name>/_analysis/
├── swebench_analysis.json
├── swebench_analysis.md
└── swebench_instances.csv
```

## 后续解析

下一步需要写 converter，把：

- `openhands_outputs/output.jsonl`
- `openhands_outputs/infer_logs/*`
- `thunderagent_profiles/step_profiles.csv`

合并成 `task.md` 里的统一 event-level trace schema。
