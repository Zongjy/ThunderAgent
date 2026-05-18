# Examples

This branch organizes examples around tool-call analysis rather than training pipelines.

For uv environment isolation, see [`ENVIRONMENTS.md`](ENVIRONMENTS.md). The
recommended setup is one environment per benchmark scaffold:
`.venv-openhands` and `.venv-webarena`.

## Scaffold

Agent frameworks that can be pointed at ThunderAgent:

| Agent | Directory | Description |
|-------|-----------|-------------|
| OpenHands | [`scaffold/openhands`](scaffold/openhands) | Software development agent scaffold used by SWE-bench runs |
| BrowserGym + WebArena-Verified | [`scaffold/browsergym_webarena`](scaffold/browsergym_webarena) | BrowserGym web-agent scaffold with ThunderAgent-backed OpenAI calls |

## Adapters

Scaffold integration glue that should stay outside upstream scaffold checkouts:

| Adapter | Directory | Description |
|---------|-----------|-------------|
| OpenHands | [`adapters/openhands`](adapters/openhands) | ThunderAgent adapter pattern for current OpenHands / SDK checkouts |
| BrowserGym | [`adapters/browsergym`](adapters/browsergym) | ThunderAgent adapter for BrowserGym/WebArena-Verified episodes |

## Benchmark

Benchmark-specific runners and analysis code:

| Benchmark | Directory | Description |
|-----------|-----------|-------------|
| SWE-bench | [`benchmark/swebench`](benchmark/swebench) | OpenHands + ThunderAgent runner and tool-call/profile analysis utilities |
| WebArena-Verified | [`benchmark/webarena`](benchmark/webarena) | BrowserGym + WebArena-Verified runner and browser action/observation analysis |
