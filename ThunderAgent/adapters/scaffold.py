"""Small scaffold-facing adapter for ThunderAgent program IDs.

The adapter keeps ThunderAgent-specific request metadata out of scaffold
checkouts. Scaffolds only need to create one program per task/instance, attach
that program ID to LLM request metadata, and release the program when done.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, MutableMapping

_PROGRAM_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "thunderagent_program_id", default=None
)


def normalize_thunderagent_base_url(base_url: str | None) -> str | None:
    """Return the ThunderAgent root URL from an OpenAI-compatible base URL."""
    if not base_url:
        return None
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")


def make_program_id(
    instance_id: str,
    *,
    scaffold: str = "scaffold",
    salt: str | None = None,
) -> str:
    """Create a short, stable-looking program ID for one scaffold task run."""
    salt = salt or f"{os.getpid()}:{time.time_ns()}"
    digest = hashlib.sha1(
        f"{scaffold}:{instance_id}:{salt}".encode("utf-8")
    ).hexdigest()
    clean_scaffold = "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in scaffold.lower()
    ).strip("-")
    prefix = clean_scaffold[:16] or "scaffold"
    return f"{prefix}-{digest[:16]}"


def current_program_id(default: str | None = None) -> str | None:
    """Return the current context-local ThunderAgent program ID."""
    return (
        _PROGRAM_ID.get()
        or os.environ.get("THUNDERAGENT_PROGRAM_ID")
        or os.environ.get("OPENHANDS_PROGRAM_ID")
        or default
    )


def inject_program_id(
    extra_body: Mapping[str, Any] | None,
    program_id: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``extra_body`` with ThunderAgent's ``program_id`` set."""
    resolved_program_id = program_id or current_program_id()
    if not resolved_program_id:
        return dict(extra_body or {})

    body = dict(extra_body or {})
    body["program_id"] = resolved_program_id
    return body


def with_program_id(llm_config: Any, program_id: str | None = None) -> Any:
    """Return an LLM config copy whose LiteLLM extra body carries ``program_id``.

    This works with Pydantic v2 models used by recent OpenHands SDK releases.
    It also supports legacy OpenHands V0 ``LLMConfig`` objects, which pass
    LiteLLM extras through ``completion_kwargs`` instead of
    ``litellm_extra_body``.
    """
    resolved_program_id = program_id or current_program_id()
    if not resolved_program_id:
        return llm_config

    if hasattr(llm_config, "completion_kwargs"):
        completion_kwargs = dict(getattr(llm_config, "completion_kwargs", None) or {})
        completion_kwargs["extra_body"] = inject_program_id(
            completion_kwargs.get("extra_body"),
            resolved_program_id,
        )

        if hasattr(llm_config, "model_copy"):
            return llm_config.model_copy(
                deep=True, update={"completion_kwargs": completion_kwargs}
            )

        if hasattr(llm_config, "copy"):
            try:
                return llm_config.copy(
                    deep=True, update={"completion_kwargs": completion_kwargs}
                )
            except TypeError:
                pass

        setattr(llm_config, "completion_kwargs", completion_kwargs)
        return llm_config

    extra_body = inject_program_id(
        getattr(llm_config, "litellm_extra_body", None),
        resolved_program_id,
    )

    if hasattr(llm_config, "model_copy"):
        return llm_config.model_copy(
            deep=True, update={"litellm_extra_body": extra_body}
        )

    if hasattr(llm_config, "copy"):
        try:
            return llm_config.copy(deep=True, update={"litellm_extra_body": extra_body})
        except TypeError:
            pass

    setattr(llm_config, "litellm_extra_body", extra_body)
    return llm_config


def patch_litellm_kwargs(
    kwargs: MutableMapping[str, Any],
    program_id: str | None = None,
) -> MutableMapping[str, Any]:
    """Inject ``extra_body.program_id`` into a LiteLLM kwargs mapping in place."""
    kwargs["extra_body"] = inject_program_id(kwargs.get("extra_body"), program_id)
    return kwargs


def release_program(
    base_url: str | None,
    program_id: str | None,
    *,
    timeout: float = 5.0,
) -> bool:
    """Release ThunderAgent router-side state for ``program_id``.

    Returns ``True`` when the release request succeeds or there is nothing to
    release, and ``False`` when ThunderAgent rejects or cannot receive it.
    """
    root_url = normalize_thunderagent_base_url(base_url)
    if not root_url or not program_id:
        return True

    request = urllib.request.Request(
        f"{root_url}/programs/release",
        data=json.dumps({"program_id": program_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


@dataclass(frozen=True)
class ThunderAgentProgram:
    """A single scaffold task/program tracked by ThunderAgent."""

    program_id: str
    base_url: str | None
    release_on_exit: bool = True

    @classmethod
    def create(
        cls,
        instance_id: str,
        *,
        base_url: str | None,
        scaffold: str = "scaffold",
        release_on_exit: bool = True,
    ) -> "ThunderAgentProgram":
        return cls(
            program_id=make_program_id(instance_id, scaffold=scaffold),
            base_url=base_url,
            release_on_exit=release_on_exit,
        )

    def llm_config(self, llm_config: Any) -> Any:
        return with_program_id(llm_config, self.program_id)

    def llm_kwargs(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        return patch_litellm_kwargs(kwargs, self.program_id)

    def release(self) -> bool:
        return release_program(self.base_url, self.program_id)


@contextlib.contextmanager
def use_program(program: ThunderAgentProgram | str) -> Iterator[str]:
    """Set a context-local ThunderAgent program ID for one task run."""
    if isinstance(program, ThunderAgentProgram):
        program_id = program.program_id
        release_on_exit = program.release_on_exit
        base_url = program.base_url
    else:
        program_id = program
        release_on_exit = False
        base_url = None

    token = _PROGRAM_ID.set(program_id)
    try:
        yield program_id
    finally:
        _PROGRAM_ID.reset(token)
        if release_on_exit:
            release_program(base_url, program_id)
