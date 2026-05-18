"""vLLM metrics parsing, storage, and client."""
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
import time

import httpx

from .metrics_base import MetricsClient

logger = logging.getLogger(__name__)

VLLM_KV_CAPACITY_LOG_RE = re.compile(
    r"GPU KV cache size:\s*(?P<size>[0-9,]+)\s+tokens"
)
VLLM_LOG_TAIL_BYTES = 1024 * 1024


@dataclass
class VLLMCacheConfig:
    """Static KV cache configuration from vLLM (fetched once at startup)."""
    block_size: int = 0          # Tokens per block
    num_gpu_blocks: int = 0      # Total GPU blocks
    total_tokens_override: int = 0
    capacity_source: str = "cache_config_info"
    
    @property
    def total_tokens_capacity(self) -> int:
        """Total KV cache capacity in tokens."""
        if self.total_tokens_override > 0:
            return self.total_tokens_override
        return self.block_size * self.num_gpu_blocks

    def set_total_tokens_override(self, value: int, source: str) -> None:
        """Set an effective capacity discovered outside cache_config_info."""
        self.total_tokens_override = value
        self.capacity_source = source
    
    @classmethod
    def from_prometheus_text(cls, text: str) -> "VLLMCacheConfig":
        """Parse cache_config_info from Prometheus metrics text."""
        config = cls()
        
        # Extract block_size and num_gpu_blocks from labels
        # vllm:cache_config_info{block_size="16",...,num_gpu_blocks="27283",...} 1.0
        match = re.search(r'vllm:cache_config_info\{([^}]+)\}', text)
        if match:
            labels = match.group(1)
            
            # Extract block_size
            bs_match = re.search(r'block_size="(\d+)"', labels)
            if bs_match:
                config.block_size = int(bs_match.group(1))
            
            # Extract num_gpu_blocks
            ngb_match = re.search(r'num_gpu_blocks="(\d+)"', labels)
            if ngb_match:
                config.num_gpu_blocks = int(ngb_match.group(1))

        return config


def parse_vllm_kv_capacity_from_log_text(text: str) -> Optional[int]:
    """Parse vLLM's effective GPU KV cache capacity from startup logs."""
    capacity = None
    for match in VLLM_KV_CAPACITY_LOG_RE.finditer(text):
        capacity = int(match.group("size").replace(",", ""))
    return capacity


def parse_positive_int(value: str, name: str) -> Optional[int]:
    """Parse a positive integer that may contain comma separators."""
    value = value.strip()
    if not value:
        return None
    try:
        parsed = int(value.replace(",", ""))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, value)
        return None
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r", name, value)
        return None
    return parsed


def first_csv_value(value: str) -> str:
    """Return the first non-empty item from a comma-separated value."""
    for item in value.split(","):
        item = item.strip()
        if item:
            return item
    return ""


@dataclass
class VLLMMetrics:
    """Parsed metrics from vLLM /metrics endpoint."""
    # Request stats
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    
    # KV Cache
    kv_cache_usage_perc: float = 0.0
    
    # Prefix cache (cumulative)
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    
    # Tokens (cumulative)
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    
    # Preemptions
    num_preemptions: int = 0
    
    # Request success counts
    request_success_stop: int = 0
    request_success_length: int = 0
    request_success_abort: int = 0
    request_success_error: int = 0
    
    # Timestamp when metrics were fetched
    timestamp: float = 0.0
    
    @property
    def prefix_cache_hit_rate(self) -> float:
        """Calculate prefix cache hit rate."""
        if self.prefix_cache_queries == 0:
            return 0.0
        return self.prefix_cache_hits / self.prefix_cache_queries
    
    @property
    def total_requests_completed(self) -> int:
        """Total completed requests."""
        return (self.request_success_stop + self.request_success_length + 
                self.request_success_abort + self.request_success_error)
    
    @classmethod
    def from_prometheus_text(cls, text: str) -> "VLLMMetrics":
        """Parse Prometheus text format into VLLMMetrics."""
        metrics = cls(timestamp=time.time())
        
        # Helper to extract gauge/counter value
        def extract_value(pattern: str) -> Optional[float]:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    return None
            return None
        
        # Current request counts
        val = extract_value(r'vllm:num_requests_running\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.num_requests_running = int(val)
        
        val = extract_value(r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.num_requests_waiting = int(val)
        
        # KV cache usage
        val = extract_value(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.kv_cache_usage_perc = val
        
        # Prefix cache (counters)
        val = extract_value(r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prefix_cache_queries = int(val)
        
        val = extract_value(r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prefix_cache_hits = int(val)
        
        # Token counts
        val = extract_value(r'vllm:prompt_tokens_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prompt_tokens_total = int(val)
        
        val = extract_value(r'vllm:generation_tokens_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.generation_tokens_total = int(val)
        
        # Preemptions
        val = extract_value(r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.num_preemptions = int(val)
        
        # Request success counts by finish reason
        for reason in ['stop', 'length', 'abort', 'error']:
            val = extract_value(rf'vllm:request_success_total\{{[^}}]*finished_reason="{reason}"[^}}]*\}}\s+([\d.]+)')
            if val is not None:
                setattr(metrics, f'request_success_{reason}', int(val))
        
        return metrics


# Keep only the most recent N metrics samples
METRICS_HISTORY_SIZE = 12


class VLLMMetricsClient(MetricsClient):
    """Client for fetching and managing metrics from a vLLM backend.
    
    Handles HTTP communication with vLLM /metrics endpoint,
    metrics history management, and shared_tokens calculation.
    """
    
    def __init__(
        self,
        url: str,
        *,
        kv_capacity_tokens: Optional[int] = None,
        log_path: Optional[str] = None,
    ):
        super().__init__(url)
        self.healthy = True
        self.metrics_history: List[VLLMMetrics] = []
        self.cache_config: Optional[VLLMCacheConfig] = None
        self.kv_capacity_tokens = (
            kv_capacity_tokens
            if kv_capacity_tokens is not None
            else parse_positive_int(
                os.environ.get("THUNDERAGENT_KV_CAPACITY_TOKENS", ""),
                "THUNDERAGENT_KV_CAPACITY_TOKENS",
            )
        )
        self.log_path = (
            log_path
            or os.environ.get("THUNDERAGENT_VLLM_LOG_PATH", "").strip()
            or first_csv_value(os.environ.get("THUNDERAGENT_VLLM_LOG_PATHS", ""))
        )
        
        # HTTP client and monitoring state
        self._client: Optional[httpx.AsyncClient] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_stop = False
    
    @property
    def metrics_url(self) -> str:
        """Prometheus metrics endpoint."""
        return f"{self.url}/metrics"
    
    @property
    def latest_metrics(self) -> Optional[VLLMMetrics]:
        """Get the most recent metrics sample."""
        return self.metrics_history[-1] if self.metrics_history else None

    def _read_kv_capacity_from_log(self) -> Optional[int]:
        """Read vLLM's effective KV capacity from the configured startup log."""
        if not self.log_path:
            return None
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - VLLM_LOG_TAIL_BYTES))
                text = f.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            logger.debug("vLLM log path does not exist yet: %s", self.log_path)
            return None
        except OSError as exc:
            logger.warning("Failed to read vLLM log %s: %s", self.log_path, exc)
            return None
        return parse_vllm_kv_capacity_from_log_text(text)

    def _apply_effective_capacity(self, config: VLLMCacheConfig) -> None:
        """Prefer vLLM's logged effective capacity over static cache_config_info."""
        log_capacity = self._read_kv_capacity_from_log()
        if log_capacity:
            fallback = config.block_size * config.num_gpu_blocks
            if fallback and fallback != log_capacity:
                logger.info(
                    "Using vLLM log KV capacity for %s: %s tokens "
                    "(cache_config_info would be %s)",
                    self.url,
                    log_capacity,
                    fallback,
                )
            config.set_total_tokens_override(log_capacity, "vllm_log")
            return

        if self.kv_capacity_tokens:
            config.set_total_tokens_override(
                self.kv_capacity_tokens,
                "configured",
            )
    
    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._monitor_task is not None
    
    # -------------------------------------------------------------------------
    # Metrics Monitoring
    # -------------------------------------------------------------------------
    
    async def start_monitoring(self, interval: float = 5.0):
        """Start background metrics monitoring."""
        if self._monitor_task is not None:
            return  # Already running
        
        self._monitor_stop = False
        self._client = httpx.AsyncClient(timeout=10.0)
        
        # Fetch cache config once at startup
        await self.fetch_cache_config()
        
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Started metrics monitoring for {self.url} (interval: {interval}s, kv_capacity: {self.cache_config.total_tokens_capacity if self.cache_config else 'unknown'} tokens)")
    
    async def stop_monitoring(self):
        """Stop background metrics monitoring."""
        if self._monitor_task is None:
            return
        
        self._monitor_stop = True
        self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass
        self._monitor_task = None
        
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info(f"Stopped metrics monitoring for {self.url}")
    
    async def _monitor_loop(self, interval: float):
        """Background loop to periodically fetch metrics."""
        while not self._monitor_stop:
            try:
                await self.fetch_metrics()
                if (
                    self.cache_config
                    and self.cache_config.capacity_source != "vllm_log"
                ):
                    self._apply_effective_capacity(self.cache_config)
            except Exception as e:
                logger.debug(f"Error fetching metrics from {self.url}: {e}")
            await asyncio.sleep(interval)
    
    # -------------------------------------------------------------------------
    # Metrics Fetching
    # -------------------------------------------------------------------------
    
    async def fetch_cache_config(self) -> bool:
        """Fetch static cache config from vLLM.
        
        Can be called independently of start_monitoring().
        Uses existing client if monitoring, otherwise creates a temporary one.
        """
        client = self._client
        close_client = False
        
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            close_client = True
        
        try:
            resp = await client.get(self.metrics_url)
            if resp.status_code == 200:
                self.cache_config = VLLMCacheConfig.from_prometheus_text(resp.text)
                self._apply_effective_capacity(self.cache_config)
                override_msg = (
                    f", effective_capacity={self.cache_config.total_tokens_override}"
                    if self.cache_config.total_tokens_override > 0
                    else ""
                )
                logger.info(
                    "Fetched cache config for %s: block_size=%s, "
                    "num_gpu_blocks=%s, total_capacity=%s, source=%s%s",
                    self.url,
                    self.cache_config.block_size,
                    self.cache_config.num_gpu_blocks,
                    self.cache_config.total_tokens_capacity,
                    self.cache_config.capacity_source,
                    override_msg,
                )
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to fetch cache config from {self.url}: {e}")
            return False
        finally:
            if close_client:
                await client.aclose()
    
    async def fetch_metrics(self) -> bool:
        """Fetch and update metrics from vLLM /metrics endpoint."""
        if not self._client:
            return False
        try:
            resp = await self._client.get(self.metrics_url)
            if resp.status_code == 200:
                metrics = VLLMMetrics.from_prometheus_text(resp.text)
                self.metrics_history.append(metrics)
                # Keep only the most recent samples
                if len(self.metrics_history) > METRICS_HISTORY_SIZE:
                    self.metrics_history = self.metrics_history[-METRICS_HISTORY_SIZE:]
                self.healthy = True
                return True
            else:
                self.healthy = False
                return False
        except Exception as e:
            logger.debug(f"Failed to fetch metrics from {self.url}: {e}")
            self.healthy = False
            return False
    
    # -------------------------------------------------------------------------
    # Calculations
    # -------------------------------------------------------------------------
    
    def calculate_shared_tokens(self, reasoning_program_tokens: int) -> int:
        """Calculate shared tokens (prefix cache savings).
        
        shared_tokens = reasoning_program_tokens - vllm_actual_used_tokens
        
        Args:
            reasoning_program_tokens: Current total tokens from REASONING programs
        
        Returns:
            Shared tokens (0 if no metrics available or negative)
        """
        if not self.latest_metrics or not self.cache_config:
            return 0
        vllm_actual_used = int(
            self.latest_metrics.kv_cache_usage_perc 
            * self.cache_config.total_tokens_capacity
        )
        return max(0, reasoning_program_tokens - vllm_actual_used)
    
    def to_dict(self) -> dict:
        """Convert metrics state to dict for API response."""
        result = {
            "healthy": self.healthy,
            "monitoring": self.is_monitoring,
        }
        # Include cache config (static)
        if self.cache_config:
            result["cache_config"] = {
                "block_size": self.cache_config.block_size,
                "num_gpu_blocks": self.cache_config.num_gpu_blocks,
                "total_tokens_capacity": self.cache_config.total_tokens_capacity,
                "capacity_source": self.cache_config.capacity_source,
            }
            if self.cache_config.total_tokens_override > 0:
                result["cache_config"]["total_tokens_override"] = (
                    self.cache_config.total_tokens_override
                )
        # Include latest metrics (dynamic)
        if self.metrics_history:
            latest = self.latest_metrics
            result["metrics"] = {
                "num_requests_running": latest.num_requests_running,
                "num_requests_waiting": latest.num_requests_waiting,
                "kv_cache_usage_perc": round(latest.kv_cache_usage_perc, 4),
                "prefix_cache_hit_rate": round(latest.prefix_cache_hit_rate, 4),
                "prefix_cache_queries": latest.prefix_cache_queries,
                "prefix_cache_hits": latest.prefix_cache_hits,
                "prompt_tokens_total": latest.prompt_tokens_total,
                "generation_tokens_total": latest.generation_tokens_total,
                "num_preemptions": latest.num_preemptions,
                "requests_completed": latest.total_requests_completed,
                "last_updated": latest.timestamp,
                "history_size": len(self.metrics_history),
            }
        return result
