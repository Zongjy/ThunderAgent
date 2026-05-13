# Examples

This branch organizes examples around tool-call analysis rather than training pipelines.

## Scaffold

Agent frameworks that can be pointed at ThunderAgent:

| Agent | Directory | Description |
|-------|-----------|-------------|
| OpenHands | [`scaffold/openhands`](scaffold/openhands) | Software development agent scaffold used by SWE-bench runs |
| mini-swe-agent | [`scaffold/mini-swe-agent`](scaffold/mini-swe-agent) | Lightweight SWE-agent scaffold with Docker-based code sandboxes |
| ToolOrchestra | [`scaffold/toolorchestra`](scaffold/toolorchestra) | Multi-tool orchestration scaffold for HLE-style runs |
| tau^3 | [`scaffold/tau3`](scaffold/tau3) | Official tau2/τ³ runner wrapper with a ThunderAgent-backed LLMAgent |

## Adapters

Scaffold integration glue that should stay outside upstream scaffold checkouts:

| Adapter | Directory | Description |
|---------|-----------|-------------|
| OpenHands | [`adapters/openhands`](adapters/openhands) | ThunderAgent adapter pattern for current OpenHands / SDK checkouts |
| tau^3 | [`adapters/tau3`](adapters/tau3) | ThunderAgent adapter for tau^3 agent scaffold |

## Benchmark

Benchmark-specific runners and analysis code:

| Benchmark | Directory | Description |
|-----------|-----------|-------------|
| SWE-bench | [`benchmark/swebench`](benchmark/swebench) | OpenHands + ThunderAgent runner and tool-call/profile analysis utilities |
| tau^3 | [`benchmark/tau3`](benchmark/tau3) | Official tau2/τ³ runner (ta default) and analysis |
