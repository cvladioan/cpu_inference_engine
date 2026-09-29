# cpu_inference_engine

Run large Mixture-of-Experts models on CPU-only x86 servers, with no GPUs.
The first target is DeepSeek-V4-Flash (284B total, 13B active).

- [`tessera/`](tessera/README.md): **Tessera**, the GPU + CPU + SSD engine for
  30-120B mixture-of-experts models on a gaming PC (RTX 4070 12 GB, 32 GB RAM):
  hot experts in VRAM, the rest on the CPU at the same time, cold ones from the
  SSD. Self-contained, ready to move into its own repository.
- [`docs/HANDOFF.md`](docs/HANDOFF.md): current status, research findings and
  next steps. Read this first when resuming work.
- [`deploy/`](deploy/README.md): **start here to run a model.** Serves DeepSeek-V4-Flash today
  with ik_llama.cpp behind an OpenAI-compatible API. Includes host checks,
  build, model download, NUMA-aware launch, benchmarks and a systemd service.
- [`docs/BOTTLENECKS.md`](docs/BOTTLENECKS.md): measured analysis of what blocks big
  models on common CPUs, and the research agenda that follows.
- [`docs/EXPERT_CACHE.md`](docs/EXPERT_CACHE.md): the explicit expert cache, an
  engine change (`engine/patches/`) that runs MoE models larger than RAM 4-6x
  faster than the OS page cache.
- [`docs/HOME_TESTS.md`](docs/HOME_TESTS.md): step-by-step tests of the expert
  cache with DeepSeek-V4-Flash on a 32 GB Windows PC (WSL2).
- [`docs/PLAN.md`](docs/PLAN.md): research, hardware sizing, performance
  estimates, and the longer-term plan for a custom engine.
- [`tools/roofline.py`](tools/roofline.py): bandwidth roofline calculator
  behind the estimates in the plan. Run `python3 tools/roofline.py`; it uses
  only the standard library.
