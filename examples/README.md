# Examples

This branch organizes examples around tool-call analysis rather than training pipelines.

## Scaffold

Agent frameworks that can be pointed at ThunderAgent:

| Agent | Directory | Description |
|-------|-----------|-------------|
| OpenHands | [`scaffold/openhands`](scaffold/openhands) | Software development agent scaffold used by SWE-bench runs |
| mini-swe-agent | [`scaffold/mini-swe-agent`](scaffold/mini-swe-agent) | Lightweight SWE-agent scaffold with Docker-based code sandboxes |
| ToolOrchestra | [`scaffold/toolorchestra`](scaffold/toolorchestra) | Multi-tool orchestration scaffold for HLE-style runs |

## Benchmark

Benchmark-specific runners and analysis code:

| Benchmark | Directory | Description |
|-----------|-----------|-------------|
| SWE-bench | [`benchmark/swebench`](benchmark/swebench) | OpenHands + ThunderAgent runner and tool-call/profile analysis utilities |
| tau-bench | [`benchmark/tau-bench`](benchmark/tau-bench) | tau-bench tool-call adapters kept without slime training code |
