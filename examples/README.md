# Examples

This branch organizes examples around tool-call analysis rather than training pipelines.

For uv environment isolation, see [`ENVIRONMENTS.md`](ENVIRONMENTS.md). The
recommended setup is one environment per benchmark scaffold:
`.venv-openhands` and `.venv-osworld`.

## Scaffold

Agent frameworks that can be pointed at ThunderAgent:

| Agent | Directory | Description |
|-------|-----------|-------------|
| OpenHands | [`scaffold/openhands`](scaffold/openhands) | Software development agent scaffold used by SWE-bench runs |
| OSWorld + Qwen3.5 | [`scaffold/osworld`](scaffold/osworld) | Thin wrapper around OSWorld's native Qwen3VL multi-env GUI runner |

## Adapters

Scaffold integration glue that should stay outside upstream scaffold checkouts:

| Adapter | Directory | Description |
|---------|-----------|-------------|
| OpenHands | [`adapters/openhands`](adapters/openhands) | ThunderAgent adapter pattern for current OpenHands / SDK checkouts |
| OSWorld | [`adapters/osworld`](adapters/osworld) | ThunderAgent adapter for OSWorld desktop episodes |

## Benchmark

Benchmark-specific runners and analysis code:

| Benchmark | Directory | Description |
|-----------|-----------|-------------|
| SWE-bench | [`benchmark/swebench`](benchmark/swebench) | OpenHands + ThunderAgent runner and tool-call/profile analysis utilities |
| OSWorld-Verified | [`benchmark/osworld_verified`](benchmark/osworld_verified) | OSWorld native Qwen3VL runner through Qwen3.5 and ThunderAgent |
