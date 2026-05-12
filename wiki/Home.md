# ThunderAgent Wiki

**ThunderAgent** is a fast and easy-to-use library for agentic LLM inference workloads. It sits between agent clients and inference engines (vLLM, SGLang) as a program-aware scheduler, improving throughput by 1.5-3.6x across diverse agentic serving workloads.

## Key Features

- **Program-Aware Scheduling**: Increases KV-cache hit rate and reduces memory imbalance across nodes through capacity-based pause/resume of agentic programs.
- **Tool-Call Lifecycle Management**: Per-program tracking for timing, token, pause, and tool-use analysis.
- **OpenAI-Compatible API**: Drop-in proxy; only requires adding `program_id` to your existing OpenAI API calls.
- **Multi-Backend Support**: Works with inference serving engines [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).
- **Profiling**: Per-program timing metrics (prefill, decode, tool-call, pause) with CSV export and live API.

## Wiki Contents

| Page | Description |
|------|-------------|
| [Architecture](Architecture.md) | System design, module structure, and data flow |
| [Getting Started](Getting-Started.md) | Installation, quick start, and integration guide |
| [Configuration](Configuration.md) | CLI arguments and runtime configuration |
| [API Reference](API-Reference.md) | REST API endpoints exposed by ThunderAgent |
| [Scheduler](Scheduler.md) | Program-aware scheduling: TR mode, pause/resume, BFD bin-packing |
| [Backend Support](Backend-Support.md) | vLLM and SGLang backend integration |
| [Profiling and Metrics](Profiling-and-Metrics.md) | Profiling, metrics monitoring, and CSV export |
| [Examples](Examples.md) | Scaffold and benchmark examples |
