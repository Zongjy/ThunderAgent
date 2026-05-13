# tau^3 ThunderAgent Adapter

Wraps `ThunderAgent.adapters` for the official tau2/τ³ runner. One official
benchmark task maps to one ThunderAgent program.

## Usage

```python
from examples.adapters.tau3 import tau3_instance

with tau3_instance(kwargs, instance_id="airline_0", base_url="http://127.0.0.1:9000") as (prog, patched_kwargs):
    response = client.chat.completions.create(**patched_kwargs)
```

The production path is `examples/scaffold/tau3/official_agent.py`, which
registers this behavior as a tau2 agent factory.
