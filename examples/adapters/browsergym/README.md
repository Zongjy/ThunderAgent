# BrowserGym ThunderAgent Adapter

This adapter keeps ThunderAgent-specific request metadata out of BrowserGym or
AgentLab code. Wrap one BrowserGym episode with `browsergym_instance()` and pass
the returned `extra_body` to an OpenAI-compatible LLM call:

```python
from examples.adapters.browsergym import browsergym_instance

llm_kwargs = {"model": "Qwen/Qwen3.5-9B"}

with browsergym_instance(
    llm_kwargs,
    instance_id="webarena.0",
    base_url="http://127.0.0.1:9000/v1",
) as (program, patched_kwargs):
    # patched_kwargs["extra_body"]["program_id"] is now set.
    ...
```

The production path in this repository is
`examples/scaffold/browsergym_webarena/run_browsergym_webarena.py`.
