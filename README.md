<!-- markdownlint-disable MD001 MD041 -->
# ThunderAgent

<p align="center">
| <a href="https://deepwiki.com/HaoKang-Timmy/ThunderAgent"><b>Wiki</b></a> | <a href="https://thunderagent.ai"><b>Blog</b></a> | <a href="https://arxiv.org/pdf/2602.13692"><b>Paper</b></a> |
</p>

---

## About
ThunderAgent is a fast and easy-to-use library for program-aware agentic inference. This branch is trimmed for analyzing tool-call behavior across different agent benchmarks.

ThunderAgent is fast with:

- Agentic program-aware scheduler that increases KV-cache hit rate and reduces memory imbalance across nodes, increasing agentic inference **throughputs 1.5-3.6x** across multiple agentic workflows.
- Tool-call lifecycle tracking for per-program profiling and benchmark analysis.

ThunderAgent is flexible and easy to use with:
- OpenAI-compatible API passthrough with only one changing, adding `Program_id` to the sending API.
- Multiple inference support for [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)
- Scaffold examples for OpenHands and OSWorld under [`examples/scaffold`](examples/scaffold).
- Benchmark analysis entry points for SWE-bench and OSWorld-Verified workloads under [`examples/benchmark`](examples/benchmark).
- Per-program profiling for total tokens, tool-use time, pause time, and request timing.

### Overview

ThunderAgent sits between agent clients and vLLM/SGLang as an OpenAI-compatible proxy. It tracks each agent program by `program_id`, schedules requests across one or more backends, and exposes profiling data that is useful for studying tool-call-heavy benchmarks.

### Inference & Evaluation Results

ThunderAgent improves vLLM throughput by **1.5–3.6×** across diverse agentic workloads including SWE-Agent and OpenHands.

## Demo

https://github.com/user-attachments/assets/751d8fc3-a9d8-482c-a90c-6fe29e53008f

## Getting Started

Install ThunderAgent from source:

```bash
git clone git@github.com:HaoKang-Timmy/ThunderAgent.git
cd ThunderAgent
pip install -e .
```
How to use?
Choose one backend you like, for example vllm.
```bash
uv pip install vllm --torch-backend=auto # install vllm

vllm serve Qwen/Qwen3-32B --port 8000 # serve a model

thunderagent --backend-type vllm --backends http://localhost:8000 --port 9000 --metrics --profile # launch ThunderAgent, make sure to send request through 9000.
```
How to embed with your own agentic workflow?

```python
# original openai sender
openai.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
          )
# ThunderAgent openai sender
extra_body = {}
extra_body["program_id"] = "unique_id"
# if you use docker for your agentic workflow
# extra_body["docker_ids"] = ["docker_id1", "docker_id2", ...]
openai.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            extra_body = extra_body
          )
```



## Contributing

We welcome and value any contributions and collaborations.
Please create a pull request.

## Citation

If you use ThunderAgent for your research, please cite our [paper](https://arxiv.org/abs/2602.13692):

```bibtex
@misc{kang2026thunderagentsimplefastprogramaware,
      title={ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System}, 
      author={Hao Kang and Ziyang Li and Xinyu Yang and Weili Xu and Yinfang Chen and Junxiong Wang and Beidi Chen and Tushar Krishna and Chenfeng Xu and Simran Arora},
      year={2026},
      eprint={2602.13692},
      archivePrefix={arXiv},
      primaryClass={cs.OS},
      url={https://arxiv.org/abs/2602.13692}, 
}
```

## Contact Us
For enterprises interested in adopting or deploying ThunderAgent at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at [hkang342@gatech.edu](mailto:hkang342@gatech.edu).

## License
This repository is available under the MIT license. See the [LICENSE.md](https://github.com/ThunderAgent-org/ThunderAgent/blob/main/LICENSE.md) file for details.
