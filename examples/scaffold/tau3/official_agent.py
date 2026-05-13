"""ThunderAgent adapter for the official tau2/τ³ LLMAgent.

This module does not reimplement the tau benchmark loop. It registers a tiny
agent factory with tau2's official registry and lets ``tau2.run_domain`` build
the official environment, user simulator, retrieval pipeline, orchestrator, and
evaluator.
"""

from __future__ import annotations

import weakref
from typing import Any

from tau2.agent.llm_agent import LLMAgent
from tau2.registry import registry

from examples.adapters.tau3.integration import prepare_kwargs


class ThunderAgentLLMAgent(LLMAgent):
    """Official tau2 LLMAgent with ThunderAgent ``program_id`` injection."""

    def __init__(
        self,
        *,
        tools,
        domain_policy: str,
        llm: str,
        llm_args: dict[str, Any] | None = None,
        task=None,
        **_: Any,
    ) -> None:
        instance_id = getattr(task, "id", None) or "unknown"
        base_url = _thunderagent_root_url(llm_args or {})
        program, patched_llm_args = prepare_kwargs(
            llm_args or {},
            instance_id=str(instance_id),
            base_url=base_url,
            scaffold="tau3-official",
        )
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=dict(patched_llm_args),
        )
        self.thunderagent_program = program
        self._release_program = weakref.finalize(self, program.release)


def _thunderagent_root_url(llm_args: dict[str, Any]) -> str | None:
    base_url = (
        llm_args.get("api_base")
        or llm_args.get("base_url")
        or llm_args.get("api_base_url")
    )
    if not isinstance(base_url, str) or not base_url:
        return None
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url.rstrip("/")


def create_thunderagent_llm_agent(tools, domain_policy, **kwargs):
    """Factory function called by the official tau2 runner."""
    return ThunderAgentLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )


def register_thunderagent_agent(name: str = "thunderagent_llm_agent") -> str:
    """Register the ThunderAgent-backed agent factory if needed."""
    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(create_thunderagent_llm_agent, name)
    return name
