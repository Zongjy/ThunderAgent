"""Adapters for integrating scaffolds with ThunderAgent."""

from ThunderAgent.adapters.scaffold import (
    ThunderAgentProgram,
    current_program_id,
    inject_program_id,
    make_program_id,
    normalize_thunderagent_base_url,
    patch_litellm_kwargs,
    release_program,
    use_program,
    with_program_id,
)

__all__ = [
    "ThunderAgentProgram",
    "current_program_id",
    "inject_program_id",
    "make_program_id",
    "normalize_thunderagent_base_url",
    "patch_litellm_kwargs",
    "release_program",
    "use_program",
    "with_program_id",
]
