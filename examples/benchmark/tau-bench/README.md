# tau-bench Tool-Call Benchmark

This directory keeps the tau-bench pieces that are useful for inspecting tool-call behavior with ThunderAgent. The old slime training scripts were intentionally removed from this branch.

Included files:

| File | Purpose |
|------|---------|
| `tau1_mock.py` | Helper for preparing tau-bench local data |
| `openai_tool_adapter.py` | Adapter that normalizes OpenAI-style tool calls into tau-bench actions |
| `sglang_tool_parser.py` | Parser utilities for SGLang/Qwen-style tool-call payloads |

Use this as the benchmark module for future tau-bench or tau3-bench style tool-call analysis. Runtime agent scaffolds should live under `examples/scaffold`, while benchmark launch/analyze scripts should live here.
