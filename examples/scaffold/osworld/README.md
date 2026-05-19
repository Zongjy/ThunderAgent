# OSWorld Agent-S Scaffold

This scaffold runs OSWorld/OSWorld-Verified tasks with OSWorld's
`DesktopEnv`, Agent-S as the only GUI agent loop, and ThunderAgent as the
OpenAI-compatible routing endpoint. The previous in-repo UI-TARS/OpenCUA loop
has been removed from this runner.

The OSWorld source tree stays external. Clone it and point
`OSWORLD_SOURCE_ROOT` at the checkout:

```bash
git clone https://github.com/xlang-ai/OSWorld.git /raid0/liyi/OSWorld
export OSWORLD_SOURCE_ROOT=/raid0/liyi/OSWorld
```

## Install

```bash
OSWORLD_SOURCE_ROOT=/raid0/liyi/OSWorld \
bash examples/scripts/setup_benchmark_env.sh osworld
```

The scaffold pins Agent-S through PyPI package `gui-agents==0.3.2` in
`requirements.txt`. Runtime strict-version checks are enabled by default and
can be disabled only with `--no-agent-s-strict-version`.

Agent-S OCR/text grounding may require the system `tesseract` binary:

```bash
sudo apt-get install -y tesseract-ocr
```

## Run One Task

Start or reuse a ThunderAgent OpenAI-compatible endpoint, then run:

```bash
python examples/scaffold/osworld/run_osworld_verified.py \
  --osworld-root /raid0/liyi/OSWorld \
  --test-config-base-dir /raid0/liyi/OSWorld/evaluation_examples \
  --test-all-meta-path /raid0/liyi/OSWorld/evaluation_examples/test_nogdrive.json \
  --base-url http://127.0.0.1:9000/v1 \
  --model ByteDance-Seed/UI-TARS-1.5-7B \
  --agent-kind agent-s \
  --domain libreoffice \
  --num-tasks 1 \
  --max-steps 100 \
  --output-dir /tmp/osworld_verified_smoke
```

Each OSWorld example receives one ThunderAgent `program_id` through
`extra_body`, and the scaffold writes unified event traces under
`<output-dir>/traces/*.jsonl`.

## Agent-S Configuration

The default Agent-S engine type is `openai`, pointed at ThunderAgent:

- `--agent-s-engine-type openai`
- `--agent-s-main-model`, `--agent-s-grounding-model`, `--agent-s-code-model`
- `--agent-s-main-base-url`, `--agent-s-grounding-base-url`,
  `--agent-s-code-base-url`
- `--agent-s-main-extra-body`, `--agent-s-grounding-extra-body`,
  `--agent-s-code-extra-body` for role-specific OpenAI extra body JSON
- `--agent-s-grounding-width` and `--agent-s-grounding-height`, normally kept
  equal to OSWorld screen size
- `--agent-s-enable-code-agent` to let Agent-S call its code agent against the
  live OSWorld environment
- `--agent-s-max-trajectory-length 8`
- `--agent-s-enable-reflection`

The runner monkey-patches Agent-S OpenAI/vLLM engines at startup so every
Agent-S model call carries the current ThunderAgent `program_id`. Use the
`openai` engine type for the normal ThunderAgent path.

## Parallel Runs

`--max-concurrency N` is supported. The runner partitions tasks across `N`
processes, and each process creates its own OSWorld `DesktopEnv` while sharing
the same ThunderAgent/vLLM endpoints.

Parallelism is limited by OSWorld provider capacity, GPU memory, vLLM
throughput, and ThunderAgent scheduling. For local VM providers, use
`--max-concurrency 1` unless you have explicitly provisioned independent VM
instances/snapshots. For AWS, make sure account quota, region capacity, and
cleanup are ready before raising concurrency.
