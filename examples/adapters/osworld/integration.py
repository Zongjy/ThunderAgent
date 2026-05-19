"""OSWorld-facing helpers for ThunderAgent program tracking.

One OSWorld task maps to one ThunderAgent program. OpenAI-compatible clients
carry that program ID through ``extra_body`` so ThunderAgent can profile and
schedule every request belonging to the same desktop episode.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, MutableMapping

from ThunderAgent.adapters import ThunderAgentProgram, use_program


def prepare_openai_kwargs(
    llm_kwargs: MutableMapping[str, Any],
    *,
    instance_id: str,
    base_url: str | None = None,
    scaffold: str = "osworld-agent-s",
) -> tuple[ThunderAgentProgram, dict[str, Any]]:
    """Return ``(program, patched_openai_kwargs)`` for one OSWorld task."""
    program = ThunderAgentProgram.create(
        instance_id=instance_id,
        scaffold=scaffold,
        base_url=base_url,
    )
    patched = dict(llm_kwargs)
    program.llm_kwargs(patched)
    return program, patched


@contextmanager
def osworld_instance(
    llm_kwargs: MutableMapping[str, Any],
    *,
    instance_id: str,
    base_url: str | None = None,
    scaffold: str = "osworld-agent-s",
) -> Iterator[tuple[ThunderAgentProgram, dict[str, Any]]]:
    """Context manager for one OSWorld/OSWorld-Verified benchmark task."""
    program, patched_kwargs = prepare_openai_kwargs(
        llm_kwargs,
        instance_id=instance_id,
        base_url=base_url,
        scaffold=scaffold,
    )
    with use_program(program):
        yield program, patched_kwargs
