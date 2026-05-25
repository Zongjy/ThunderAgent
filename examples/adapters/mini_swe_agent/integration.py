"""mini-swe-agent-facing helpers for ThunderAgent program tracking."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from ThunderAgent.adapters import ThunderAgentProgram, inject_program_id, use_program


def infer_model_base_url(model_config: Mapping[str, Any] | None) -> str | None:
    """Infer the OpenAI-compatible base URL from a mini-swe-agent model config."""
    if not model_config:
        return None

    for key in ("base_url", "api_base"):
        value = model_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    model_kwargs = model_config.get("model_kwargs")
    if isinstance(model_kwargs, Mapping):
        for key in ("base_url", "api_base"):
            value = model_kwargs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def prepare_model_config(
    model_config: Mapping[str, Any],
    *,
    instance_id: str,
    scaffold: str = "mini-swe-agent",
    base_url: str | None = None,
) -> tuple[ThunderAgentProgram, dict[str, Any]]:
    """Return ``(program, patched_model_config)`` for one SWE-bench task.

    mini-swe-agent model classes receive arbitrary config fields from YAML. We
    set both ``program_id`` and ``model_kwargs.extra_body.program_id`` so this
    works with the local ThunderAgent VLLM model and with LiteLLM-style models.
    """
    patched = copy.deepcopy(dict(model_config))
    program = ThunderAgentProgram.create(
        instance_id=instance_id,
        scaffold=scaffold,
        base_url=base_url or infer_model_base_url(patched),
    )

    patched["program_id"] = program.program_id
    model_kwargs = patched.setdefault("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        model_kwargs = dict(model_kwargs)
        patched["model_kwargs"] = model_kwargs
    model_kwargs["extra_body"] = inject_program_id(
        model_kwargs.get("extra_body"),
        program.program_id,
    )
    return program, patched


@contextmanager
def mini_swe_agent_instance(
    model_config: Mapping[str, Any],
    *,
    instance_id: str,
    scaffold: str = "mini-swe-agent",
    base_url: str | None = None,
) -> Iterator[tuple[ThunderAgentProgram, dict[str, Any]]]:
    """Context manager for one mini-swe-agent benchmark task."""
    program, patched_model_config = prepare_model_config(
        model_config,
        instance_id=instance_id,
        scaffold=scaffold,
        base_url=base_url,
    )
    with use_program(program):
        yield program, patched_model_config
