# Examples

ThunderAgent examples are split into agent scaffolds and benchmark modules.

## Scaffold

Agent frameworks that can be routed through ThunderAgent:

| Agent | Directory | Description |
|-------|-----------|-------------|
| **OpenHands** | [`scaffold/openhands`](../examples/scaffold/openhands) | General software development and science discovery agent |
| **AgentLab + WebArena** | [`scaffold/agentlab_webarena`](../examples/scaffold/agentlab_webarena) | AgentLab WebArena study scaffold with ThunderAgent-backed OpenAI calls |
| **OSWorld + Agent-S** | [`scaffold/osworld`](../examples/scaffold/osworld) | OSWorld desktop GUI scaffold using pinned Agent-S as the agent loop |

### Quick Example: OpenHands

```bash
# 1. Start vLLM backend
vllm serve Qwen/Qwen3-32B --port 8000

# 2. Start ThunderAgent
thunderagent --backend-type vllm --backends http://localhost:8000 --port 9000 --router tr --metrics --profile

# 3. Run OpenHands pointing to ThunderAgent
cd examples/scaffold/openhands
# Follow README.md for setup and execution
```

## Benchmark

Benchmark modules contain launch scripts, adapters, and analysis code for studying tool-call behavior:

| Benchmark | Directory | Description |
|-----------|-----------|-------------|
| **SWE-bench** | [`benchmark/swebench`](../examples/benchmark/swebench) | OpenHands + ThunderAgent runner and tool-call/profile analysis utilities |
| **WebArena** | [`benchmark/webarena`](../examples/benchmark/webarena) | AgentLab + WebArena runner and ThunderAgent profile capture |
| **OSWorld-Verified** | [`benchmark/osworld_verified`](../examples/benchmark/osworld_verified) | OSWorld + Agent-S runner and computer-use trace analysis |
