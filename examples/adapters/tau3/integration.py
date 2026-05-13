"""tau^3-facing helpers for ThunderAgent program tracking.

Import this module from a tau^3 benchmark runner or agent loop. One benchmark
task maps to one ThunderAgent program, and the program ID is carried in
LiteLLM's ``extra_body``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, MutableMapping

from ThunderAgent.adapters import ThunderAgentProgram, use_program


def prepare_kwargs(
    llm_kwargs: MutableMapping[str, Any],
    *,
    instance_id: str,
    base_url: str | None = None,
    scaffold: str = "tau3",
) -> tuple[ThunderAgentProgram, MutableMapping[str, Any]]:
    """Return ``(program, patched_llm_kwargs)`` for one tau^3 task."""
    program = ThunderAgentProgram.create(
        instance_id=instance_id,
        scaffold=scaffold,
        base_url=base_url,
    )
    patched = dict(llm_kwargs)
    program.llm_kwargs(patched)
    return program, patched


@contextmanager
def tau3_instance(
    llm_kwargs: MutableMapping[str, Any],
    *,
    instance_id: str,
    base_url: str | None = None,
    scaffold: str = "tau3",
) -> Iterator[tuple[ThunderAgentProgram, MutableMapping[str, Any]]]:
    """Context manager for one tau^3 benchmark task."""
    program, patched_kwargs = prepare_kwargs(
        llm_kwargs, instance_id=instance_id, base_url=base_url, scaffold=scaffold
    )
    with use_program(program):
        yield program, patched_kwargs
