# cpu_inference_engine

A plan for a CPU-only inference engine for large Mixture-of-Experts models,
such as DeepSeek-V4-Flash (284B total, 13B active) and Qwen3.5-122B-A10B, on
x86 servers without GPUs.

- [`docs/PLAN.md`](docs/PLAN.md): research, hardware sizing, engine
  architecture, roadmap and risks.
- [`tools/roofline.py`](tools/roofline.py): bandwidth roofline calculator
  behind the performance estimates in the plan. Run
  `python3 tools/roofline.py`; it uses only the standard library.
