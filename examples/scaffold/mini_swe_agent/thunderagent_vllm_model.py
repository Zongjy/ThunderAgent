"""OpenAI-compatible model for mini-swe-agent through ThunderAgent."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any, Literal

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ThunderAgent.adapters import inject_program_id
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

logger = logging.getLogger("thunderagent_vllm_model")


class ThunderAgentVLLMModelConfig(BaseModel):
    model_name: str
    base_url: str = "http://127.0.0.1:9000/v1"
    api_key: str = "EMPTY"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    program_id: str | None = None
    stream: bool = False
    timeout: float | None = None
    max_completion_tokens: int = 2048
    step_limit: int = 0
    cost_tracking: Literal["default", "ignore_errors"] = "ignore_errors"
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""

    model_config = ConfigDict(extra="allow")


class ContextLengthExceededError(Exception):
    """Raised when the model context length is exceeded."""


class ThunderAgentVLLMModel:
    """mini-swe-agent model class that sends requests through ThunderAgent."""

    def __init__(
        self,
        *,
        config_class: Callable = ThunderAgentVLLMModelConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    @staticmethod
    def _filter_openai_params(params: dict[str, Any]) -> dict[str, Any]:
        valid_params = {
            "temperature",
            "top_p",
            "n",
            "stream",
            "stop",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "seed",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "logprobs",
            "top_logprobs",
            "stream_options",
        }
        filtered = {key: value for key, value in params.items() if key in valid_params}
        dropped = set(params) - set(filtered)
        if dropped:
            logger.debug("Dropped non-OpenAI chat params: %s", sorted(dropped))
        return filtered

    def _extra_body(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_extra_body = params.pop("extra_body", None)
        if raw_extra_body is None:
            extra_body: dict[str, Any] = {}
        elif isinstance(raw_extra_body, dict):
            extra_body = dict(raw_extra_body)
        else:
            raise TypeError("model_kwargs.extra_body must be a mapping when set")

        if self.config.step_limit > 0:
            extra_body.setdefault("is_last_step", (self.n_calls + 1) >= self.config.step_limit)
        extra_body.setdefault("ignore_eos", False)
        return inject_program_id(extra_body, self.config.program_id)

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        return [{key: value for key, value in message.items() if key != "extra"} for message in messages]

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                KeyboardInterrupt,
                ContextLengthExceededError,
            )
        ),
    )
    def _query(self, messages: list[dict], **kwargs: Any):
        try:
            params = dict(self.config.model_kwargs)
            params.update(kwargs)
            params.setdefault("stream", self.config.stream)
            if "max_completion_tokens" not in params and "max_tokens" not in params:
                params["max_completion_tokens"] = self.config.max_completion_tokens

            extra_body = self._extra_body(params)
            request_params = self._filter_openai_params(params)
            request_params.setdefault("tools", [BASH_TOOL])

            return self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                extra_body=extra_body,
                **request_params,
            )
        except BadRequestError as exc:
            message = str(exc)
            if "context length" in message.lower() or "maximum context length" in message.lower():
                raise ContextLengthExceededError(message) from exc
            raise

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        self.n_calls += 1

        use_stream = kwargs.get("stream", self.config.stream)
        if use_stream:
            response = self._consume_stream(response)

        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
            "streamed": bool(use_stream),
        }
        return message

    def _consume_stream(self, stream: Any) -> Any:
        """Fold OpenAI streaming chunks into a minimal ChatCompletion-like object."""
        from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

        content_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        finish_reason = None
        response_id = ""
        created = int(time.time())
        model = self.config.model_name

        for chunk in stream:
            response_id = getattr(chunk, "id", response_id) or response_id
            created = getattr(chunk, "created", created) or created
            model = getattr(chunk, "model", model) or model
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", finish_reason) or finish_reason
            delta = choice.delta
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
            for tool_delta in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(tool_delta, "index", 0) or 0)
                part = tool_call_parts.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if getattr(tool_delta, "id", None):
                    part["id"] += tool_delta.id
                if getattr(tool_delta, "type", None):
                    part["type"] = tool_delta.type
                function = getattr(tool_delta, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        part["function"]["name"] += function.name
                    if getattr(function, "arguments", None):
                        part["function"]["arguments"] += function.arguments

        tool_calls = [
            ChatCompletionMessageToolCall.model_validate(part)
            for _, part in sorted(tool_call_parts.items())
            if part.get("id") and part.get("function", {}).get("name")
        ]
        return ChatCompletion.model_validate(
            {
                "id": response_id or f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": "".join(content_parts) or None,
                            "tool_calls": [tool_call.model_dump() for tool_call in tool_calls] or None,
                        },
                    }
                ],
                "usage": None,
            }
        )

    def _calculate_cost(self, response: Any) -> dict[str, float]:
        return {"cost": 0.0}

    def _parse_actions(self, response: Any) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs: Any) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump() | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
