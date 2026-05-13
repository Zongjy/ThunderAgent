"""ThunderAgent - VLLM proxy with program state tracking."""

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "BackendState",
    "ProgramState",
    "ProgramStatus",
    "MultiBackendRouter",
]

__version__ = "0.2.0"


def __getattr__(name: str):
    """Load heavier runtime modules only when callers ask for them."""
    if name in {"Config", "get_config", "set_config"}:
        from .config import Config, get_config, set_config

        return {
            "Config": Config,
            "get_config": get_config,
            "set_config": set_config,
        }[name]
    if name == "BackendState":
        from .backend import BackendState

        return BackendState
    if name in {"ProgramState", "ProgramStatus"}:
        from .program import ProgramState, ProgramStatus

        return {"ProgramState": ProgramState, "ProgramStatus": ProgramStatus}[name]
    if name == "MultiBackendRouter":
        from .scheduler import MultiBackendRouter

        return MultiBackendRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
