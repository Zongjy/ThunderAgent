"""Lightweight per-task timing traces for OSWorld wrapper runs."""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


_CURRENT_TRACE: contextvars.ContextVar["TaskTrace | None"] = contextvars.ContextVar(
    "osworld_timing_trace", default=None
)
_CURRENT_STEP_ID: contextvars.ContextVar[int | str | None] = contextvars.ContextVar(
    "osworld_timing_step_id", default=None
)

MAX_METADATA_STRING_CHARS = 2_000


class Span:
    """Mutable span state yielded to instrumented call sites."""

    def __init__(self, metadata: Mapping[str, Any] | None = None) -> None:
        self.metadata: dict[str, Any] = dict(metadata or {})

    def update(self, **metadata: Any) -> None:
        self.metadata.update(metadata)


class NullSpan(Span):
    """No-op span used when no task trace is active."""

    def update(self, **metadata: Any) -> None:
        return None


class TaskTrace:
    """Append-only JSONL timing trace for one OSWorld task."""

    def __init__(self, *, task_id: str, program_id: str, result_dir: str | Path) -> None:
        self.task_id = task_id
        self.program_id = program_id
        self.result_dir = Path(result_dir)
        self.path = self.result_dir / "timings.jsonl"
        self._lock = threading.Lock()
        self._disabled = False
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def write(self, event: Mapping[str, Any]) -> None:
        if self._disabled:
            return
        payload = {
            "schema_version": 1,
            "event_type": "span",
            "task_id": self.task_id,
            "program_id": self.program_id,
            "pid": os.getpid(),
            **dict(event),
        }
        try:
            line = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
        except OSError:
            self._disabled = True

    @contextmanager
    def span(
        self,
        name: str,
        *,
        step_id: int | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[Span]:
        parent_step_id = _CURRENT_STEP_ID.get()
        resolved_step_id = step_id if step_id is not None else parent_step_id
        step_token = _CURRENT_STEP_ID.set(resolved_step_id) if step_id is not None else None

        span = Span(metadata)
        start_ts = time.time()
        start_perf = time.perf_counter()
        error: str | None = None
        try:
            yield span
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end_ts = time.time()
            self.write(
                {
                    "span": name,
                    "step_id": resolved_step_id,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "duration_s": time.perf_counter() - start_perf,
                    "metadata": span.metadata,
                    "error": error,
                }
            )
            if step_token is not None:
                _CURRENT_STEP_ID.reset(step_token)


def current_trace() -> TaskTrace | None:
    return _CURRENT_TRACE.get()


def set_current_trace(trace: TaskTrace | None) -> contextvars.Token[TaskTrace | None]:
    return _CURRENT_TRACE.set(trace)


def reset_current_trace(token: contextvars.Token[TaskTrace | None]) -> None:
    _CURRENT_TRACE.reset(token)


@contextmanager
def maybe_span(
    trace: TaskTrace | None,
    name: str,
    *,
    step_id: int | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    if trace is None:
        yield NullSpan()
        return
    with trace.span(name, step_id=step_id, metadata=metadata) as span:
        yield span


def summarize_text(value: Any, *, limit: int = 240) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_METADATA_STRING_CHARS:
            return value
        return f"{value[: MAX_METADATA_STRING_CHARS - 3]}..."
    if isinstance(value, bytes):
        return {"bytes_len": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return summarize_text(value)
