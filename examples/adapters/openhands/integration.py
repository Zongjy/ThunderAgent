"""OpenHands-facing helpers for ThunderAgent program tracking.

Import this module from an OpenHands benchmark runner instead of patching the
OpenHands source tree. It is intentionally tiny: one benchmark instance maps to
one ThunderAgent program, and the program ID is carried in LiteLLM's
``extra_body``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ThunderAgent.adapters import ThunderAgentProgram, use_program


def prepare_llm_config(
    llm_config: Any,
    *,
    instance_id: str,
    scaffold: str = "openhands",
) -> tuple[ThunderAgentProgram, Any]:
    """Return ``(program, llm_config_with_program_id)`` for one instance."""
    program = ThunderAgentProgram.create(
        instance_id=instance_id,
        scaffold=scaffold,
        base_url=getattr(llm_config, "base_url", None),
    )
    return program, program.llm_config(llm_config)


@contextmanager
def openhands_instance(
    llm_config: Any,
    *,
    instance_id: str,
    scaffold: str = "openhands",
) -> Iterator[tuple[ThunderAgentProgram, Any]]:
    """Context manager for one OpenHands benchmark task.

    The yielded LLM config is a copy when the config supports Pydantic-style
    copying, so callers can safely pass it into current OpenHands SDK or
    benchmark runners without mutating shared global config.
    """
    program, patched_llm_config = prepare_llm_config(
        llm_config, instance_id=instance_id, scaffold=scaffold
    )
    with use_program(program):
        yield program, patched_llm_config
