# Examples

This branch organizes examples around tool-call analysis rather than training pipelines.

For uv environment isolation, see [`ENVIRONMENTS.md`](ENVIRONMENTS.md). The
recommended setup is one environment per benchmark scaffold:
`.venv-openhands`, `.venv-webarena`, and `.venv-osworld`.

## Scaffold

Agent frameworks that can be pointed at ThunderAgent:

| Agent | Directory | Description |
|-------|-----------|-------------|
| OpenHands | [`scaffold/openhands`](scaffold/openhands) | Software development agent scaffold used by SWE-bench runs |
| AgentLab + WebArena | [`scaffold/agentlab_webarena`](scaffold/agentlab_webarena) | AgentLab WebArena study scaffold with ThunderAgent-backed OpenAI calls |
| OSWorld + Agent-S | [`scaffold/osworld`](scaffold/osworld) | OSWorld desktop scaffold using pinned Agent-S as the GUI agent loop |

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
| WebArena | [`benchmark/webarena`](benchmark/webarena) | AgentLab + WebArena runner and ThunderAgent profile capture |
| OSWorld-Verified | [`benchmark/osworld_verified`](benchmark/osworld_verified) | OSWorld + Agent-S runner and computer-use action/observation analysis |
