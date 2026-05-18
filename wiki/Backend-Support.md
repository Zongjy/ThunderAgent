# Backend Support

ThunderAgent supports two inference serving backends: vLLM and SGLang. Each backend type has a dedicated metrics client that handles communication, metrics parsing, and capacity discovery.

## Supported Backend Types

| Backend | `--backend-type` | Type | Metrics Format | Capacity Source |
|---------|-----------------|------|----------------|-----------------|
| [vLLM](https://github.com/vllm-project/vllm) | `vllm` | Inference serving engine | Prometheus text (`/metrics`) + startup log | Effective GPU KV cache size from `GPU KV cache size: ... tokens`, with `cache_config_info` as fallback |
| [SGLang](https://github.com/sgl-project/sglang) | `sglang` | Inference serving engine | Prometheus text (`/metrics`) + JSON (`/get_server_info`) | `max_total_num_tokens` from server info |

## Architecture

All metrics clients implement the `MetricsClient` abstract interface:

```python
class MetricsClient(ABC):
    async def start_monitoring(self, interval: float) -> None
    async def stop_monitoring(self) -> None
    async def fetch_metrics(self) -> bool
    async def fetch_cache_config(self) -> bool
    def calculate_shared_tokens(self, reasoning_program_tokens: int) -> int
    def to_dict(self) -> dict
```

`BackendState` wraps a `MetricsClient` and adds program tracking, capacity checks, and token aggregation.

## vLLM Backend

ThunderAgent parses Prometheus text from vLLM's `/metrics` endpoint, including request counts, KV-cache utilization, prefix-cache hits, prompt/generation token totals, and preemption counters.

For scheduling capacity, ThunderAgent prefers the effective GPU KV cache size printed in the vLLM startup log (`GPU KV cache size: ... tokens`). This is important for hybrid models where `block_size * num_gpu_blocks` from `vllm:cache_config_info` does not represent the effective token capacity. Pass log paths with `--vllm-log-paths` or `THUNDERAGENT_VLLM_LOG_PATHS`, aligned by backend index. `--kv-capacity-tokens` / `THUNDERAGENT_KV_CAPACITY_TOKENS` is still available as a manual fallback.

## SGLang Backend

ThunderAgent combines SGLang's `/metrics` Prometheus endpoint with `/get_server_info` JSON capacity information. Capacity is read from `max_total_num_tokens` when available, with a fallback to internal state token capacity.

## Metrics Monitoring

When `--metrics` is enabled, each backend starts a background monitoring loop:

1. Fetch cache config once at startup
2. Poll `/metrics` every `--metrics-interval` seconds
3. Store recent metrics samples in a history ring buffer
4. Update backend health based on successful fetches

Metrics are used by the scheduler, `/metrics`, and `/health`.
