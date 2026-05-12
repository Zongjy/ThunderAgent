# Examples

ThunderAgent examples are split into agent scaffolds and benchmark modules.

## Scaffold

Agent frameworks that can be routed through ThunderAgent:

| Agent | Directory | Description |
|-------|-----------|-------------|
| **OpenHands** | [`scaffold/openhands`](../examples/scaffold/openhands) | General software development and science discovery agent |
| **mini-swe-agent** | [`scaffold/mini-swe-agent`](../examples/scaffold/mini-swe-agent) | Lightweight SWE-agent framework with Docker-based code sandboxes |
| **ToolOrchestra** | [`scaffold/toolorchestra`](../examples/scaffold/toolorchestra) | Multi-tool orchestration workflow |

### Quick Example: mini-swe-agent

```bash
# 1. Start vLLM backend
vllm serve Qwen/Qwen3-32B --port 8000

# 2. Start ThunderAgent
thunderagent --backend-type vllm --backends http://localhost:8000 --port 9000 --router tr --metrics --profile

# 3. Run mini-swe-agent pointing to ThunderAgent
cd examples/scaffold/mini-swe-agent
# Follow README.md for setup and execution
```

## Benchmark

Benchmark modules contain launch scripts, adapters, and analysis code for studying tool-call behavior:

| Benchmark | Directory | Description |
|-----------|-----------|-------------|
| **SWE-bench** | [`benchmark/swebench`](../examples/benchmark/swebench) | OpenHands + ThunderAgent runner and tool-call/profile analysis utilities |
| **tau-bench** | [`benchmark/tau-bench`](../examples/benchmark/tau-bench) | tau-bench tool-call adapters without slime training code |
