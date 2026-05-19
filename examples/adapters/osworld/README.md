# OSWorld ThunderAgent Adapter

This adapter keeps ThunderAgent-specific request metadata out of OSWorld code.
Wrap one OSWorld task with `osworld_instance()` and pass the returned
`extra_body` into each OpenAI-compatible vision chat completion:

```python
from examples.adapters.osworld import osworld_instance

llm_kwargs = {"model": "Qwen/Qwen3.5-9B", "temperature": 0}

with osworld_instance(
    llm_kwargs,
    instance_id="libreoffice/abc123",
    base_url="http://127.0.0.1:9000/v1",
) as (program, patched_kwargs):
    # patched_kwargs["extra_body"]["program_id"] is now set.
    ...
```

The production path in this repository is
`examples/scaffold/osworld/run_osworld_verified.py`.
