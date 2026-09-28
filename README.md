# cpu_inference_engine

Run large Mixture-of-Experts models on CPU-only x86 servers, with no GPUs.
The first target is DeepSeek-V4-Flash (284B total, 13B active).

- [`deploy/`](deploy/README.md): **start here.** Serves DeepSeek-V4-Flash today
  with ik_llama.cpp behind an OpenAI-compatible API. Includes host checks,
  build, model download, NUMA-aware launch, benchmarks and a systemd service.
- [`docs/PLAN.md`](docs/PLAN.md): research, hardware sizing, performance
  estimates, and the longer-term plan for a custom engine.
- [`tools/roofline.py`](tools/roofline.py): bandwidth roofline calculator
  behind the estimates in the plan. Run `python3 tools/roofline.py`; it uses
  only the standard library.
