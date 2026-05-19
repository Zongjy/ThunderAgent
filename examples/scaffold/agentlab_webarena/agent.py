"""AgentLab agent glue for routing WebArena LLM calls through ThunderAgent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import agentlab.llm.tracking as tracking
import openai
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.chat_api import ChatModel, RetryError, handle_error
from agentlab.llm.llm_utils import AIMessage
from openai import NOT_GIVEN, OpenAI

from ThunderAgent.adapters import ThunderAgentProgram

logger = logging.getLogger(__name__)


def normalize_openai_base_url(base_url: str | None) -> str | None:
    """Return an OpenAI-compatible endpoint URL."""
    if not base_url:
        return None
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


class ThunderAgentChatModel(ChatModel):
    """AgentLab ChatModel that sends ``extra_body.program_id`` to ThunderAgent."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        max_retry: int = 4,
        min_retry_wait_time: int = 5,
        log_probs: bool = False,
        program_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = normalize_openai_base_url(base_url)
        self.program_id = program_id
        self.extra_body = dict(extra_body or {})
        super().__init__(
            model_name=model_name,
            api_key=api_key or "EMPTY",
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else NOT_GIVEN,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=OpenAI,
            client_args={"base_url": self.base_url},
            log_probs=log_probs,
        )

    def set_program_id(self, program_id: str | None) -> None:
        self.program_id = program_id

    def _extra_body(self) -> dict[str, Any] | None:
        body = dict(self.extra_body)
        if self.program_id:
            body["program_id"] = self.program_id
        return body or None

    def __call__(
        self,
        messages: list[dict],
        n_samples: int = 1,
        temperature: float | None = None,
    ) -> dict:
        self.retries = 0
        self.success = False
        self.error_types = []

        if hasattr(messages, "to_openai"):
            messages = messages.to_openai()

        completion = None
        error_type = "unknown"
        for itr in range(self.max_retry):
            self.retries += 1
            resolved_temperature = self.temperature if temperature is None else temperature
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "n": n_samples,
                    "temperature": resolved_temperature,
                    "max_completion_tokens": self.max_tokens,
                    "logprobs": self.log_probs,
                }
                extra_body = self._extra_body()
                if extra_body:
                    request_kwargs["extra_body"] = extra_body

                completion = self.client.chat.completions.create(**request_kwargs)
                self.success = True
                break
            except openai.OpenAIError as exc:
                error_type = handle_error(
                    exc,
                    itr,
                    self.min_retry_wait_time,
                    self.max_retry,
                )
                self.error_types.append(error_type)

        if not completion:
            raise RetryError(
                "Failed to get a response from the API after "
                f"{self.max_retry} retries\nLast error: {error_type}"
            )

        usage = completion.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance,
            tracking.LLMTracker,
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        if n_samples == 1:
            res = AIMessage(completion.choices[0].message.content or "")
            if self.log_probs:
                res["log_probs"] = completion.choices[0].log_probs
            return res

        return [AIMessage(choice.message.content or "") for choice in completion.choices]


@dataclass
class ThunderAgentModelArgs(BaseModelArgs):
    """Serializable AgentLab model args for a ThunderAgent OpenAI endpoint."""

    base_url: str = "http://127.0.0.1:9000/v1"
    api_key: str = "EMPTY"
    program_id: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_retry: int = 4
    min_retry_wait_time: int = 5

    def make_model(self) -> ThunderAgentChatModel:
        return ThunderAgentChatModel(
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            max_retry=self.max_retry,
            min_retry_wait_time=self.min_retry_wait_time,
            log_probs=self.log_probs,
            program_id=self.program_id,
            extra_body=self.extra_body,
        )


class ThunderAgentGenericAgent(GenericAgent):
    """GenericAgent with one ThunderAgent program per AgentLab task."""

    def __init__(
        self,
        chat_model_args: ThunderAgentModelArgs,
        flags,
        max_retry: int = 4,
    ) -> None:
        self._program: ThunderAgentProgram | None = None
        self._program_released = True
        self.task_name: str | None = None
        super().__init__(
            chat_model_args=chat_model_args,
            flags=flags,
            max_retry=max_retry,
        )

    def set_task_name(self, task_name: str) -> None:
        self.release_program()
        self.task_name = task_name
        self._program = ThunderAgentProgram.create(
            instance_id=task_name,
            scaffold="agentlab-webarena",
            base_url=self.chat_model_args.base_url,
        )
        self._program_released = False
        if hasattr(self.chat_llm, "set_program_id"):
            self.chat_llm.set_program_id(self._program.program_id)
        logger.info("ThunderAgent program for %s: %s", task_name, self._program.program_id)

    @property
    def program_id(self) -> str | None:
        if self._program is None:
            return None
        return self._program.program_id

    def release_program(self) -> bool:
        if self._program is None or self._program_released:
            return True
        released = self._program.release()
        self._program_released = True
        if hasattr(self.chat_llm, "set_program_id"):
            self.chat_llm.set_program_id(None)
        if not released:
            logger.warning("ThunderAgent program release failed: %s", self._program.program_id)
        return released

    def get_action(self, obs):
        action, agent_info = super().get_action(obs)
        if agent_info is not None:
            agent_info.extra_info = dict(agent_info.extra_info or {})
            agent_info.extra_info["program_id"] = self.program_id

        if action is None or self._is_terminal_action(action):
            self.release_program()
        return action, agent_info

    @staticmethod
    def _is_terminal_action(action: Any) -> bool:
        text = str(action or "").strip()
        return text.startswith("send_msg_to_user(") or text.startswith("report_infeasible(")

    def close(self) -> bool:
        return self.release_program()

    def __del__(self) -> None:
        try:
            self.release_program()
        except Exception:
            pass


@dataclass
class ThunderAgentGenericAgentArgs(GenericAgentArgs):
    """AgentLab args that instantiate ``ThunderAgentGenericAgent``."""

    chat_model_args: ThunderAgentModelArgs = None

    def make_agent(self) -> ThunderAgentGenericAgent:
        return ThunderAgentGenericAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
        )

