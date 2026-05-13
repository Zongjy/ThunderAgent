# tau^3 Official Scaffold

Wrapper for the official `sierra-research/tau2-bench` τ³ runner. The benchmark
loop, domains, user simulator, retrieval pipeline, and evaluator stay in the
official package. This scaffold only registers a ThunderAgent-backed
`LLMAgent` factory that injects `extra_body.program_id` into LiteLLM requests.

## Setup

The official τ³ package currently requires Python `>=3.12`.

```bash
bash examples/scripts/setup_benchmark_env.sh tau3
```

When running from this ThunderAgent checkout, `examples/benchmark/tau3/run_tau3_ta_default.sh`
can prepend `/raid0/liyi/tau2-bench/src` to `PYTHONPATH` via `TAU2_SOURCE_ROOT`
and uses `/raid0/liyi/tau2-bench/data` as `TAU2_DATA_DIR` by default when that
local checkout exists.

## Run Official Text Benchmark

```bash
python run_official.py \
  --domain airline \
  --num-tasks 3 \
  --agent-llm openai/Qwen/Qwen3.5-9B \
  --agent-base-url http://127.0.0.1:9000/v1 \
  --user-llm openai/Qwen/Qwen3.5-9B \
  --user-base-url http://127.0.0.1:4747/v1 \
  --save-to tau3_smoke \
  --output-dir ../../benchmark/tau3/runs/tau3_smoke/official
```

Use `--domain banking_knowledge --retrieval-config bm25` for the official
knowledge benchmark path.
