# OpenHands + ThunderAgent Adapter

This directory keeps the ThunderAgent integration outside the OpenHands source
tree. The goal is to let OpenHands move independently while ThunderAgent keeps a
stable scaffold-facing contract.

## Recommended Layout

Use OpenHands as an external checkout or submodule, and keep all ThunderAgent
glue in this repository:

```bash
git submodule add -b main https://github.com/OpenHands/OpenHands.git third_party/openhands
git submodule add -b main https://github.com/OpenHands/benchmarks.git third_party/openhands-benchmarks
```

For fully reproducible experiments, pin the submodule commit after each run. For
continuous scaffold updates, prefer a normal external clone plus a lock file that
records the tested OpenHands and benchmarks commits.

## Version Note

As of 2026-05-12, OpenHands `1.7.0` is newer than `1.6.0`. OpenHands has moved
much of the current agent path toward the Software Agent SDK, while
`openhands/llm/llm.py` is legacy V0 code. New integrations should therefore
prefer SDK/benchmark-level injection over patching `openhands/llm/llm.py`.

## New OpenHands / SDK Integration

Create one ThunderAgent program per benchmark instance, copy the OpenHands LLM
config with `litellm_extra_body.program_id`, run the instance, then release the
program:

```python
from examples.adapters.openhands.integration import openhands_instance

with openhands_instance(llm_config, instance_id=instance_id) as (program, llm_config):
    # Pass llm_config into the OpenHands benchmark / SDK runner.
    # program.program_id is available for benchmark metadata.
    ...
```

This is the preferred path for `OpenHands/benchmarks` and the Software Agent SDK
because the program ID lives on the per-instance LLM config instead of a global
environment variable.

## Legacy OpenHands 1.2.x Integration

The vendored 1.2.x scaffold in `examples/scaffold/openhands` currently has a
small source patch in `openhands/llm/llm.py`. If you keep using that path, the
patch can be reduced to this generic adapter call:

```python
from ThunderAgent.adapters.scaffold import patch_litellm_kwargs

patch_litellm_kwargs(kwargs)
```

Set the current program with:

```python
from ThunderAgent.adapters import ThunderAgentProgram, use_program

program = ThunderAgentProgram.create(
    instance_id=str(instance.instance_id),
    scaffold="openhands",
    base_url=metadata.llm_config.base_url,
)

with use_program(program):
    ...
```

The legacy environment-variable approach still works, but context-local state is
safer when a scaffold runs multiple instances in one process.

## Submodule Or Not?

Submodule is a good default for this branch if you want reproducible benchmark
history without committing the whole OpenHands tree. It keeps upstream code out
of ThunderAgent diffs and makes updates explicit.

Use a normal external clone if you want to track OpenHands head daily. In that
mode, record tested commits in a small lock file and never patch the checkout
directly; keep all glue here.
