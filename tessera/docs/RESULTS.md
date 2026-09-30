# Tessera: measured results

## 2026-09-30: Qwen3-Next-80B-A3B on a CPU-only VM (first real model)

**Machine:** VMware VM on a Xeon Gold 6242R host.
- 8 vCPUs, AVX2 only (the host's AVX-512 is hidden by EVC).
- 62 GB RAM, 44.8 GB/s measured with `tools/membw.c`.
- No GPU.
- vSAN disk, about 125 MB/s.

**Model and settings:**
- Model: `unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF`, UD-Q3_K_XL, 33.2 GiB, one file. Architecture `qwen3next`:
  512 experts x 48 layers, 10 routed per token.
- Tessera (engine patches 0001-0003) in CPU-only mode (`scripts/serve.sh`):
  - 32K context, q8_0 KV cache, 6 threads;
  - all 31.7 GiB of experts in RAM, so no SSD tier.

| Request | Output | Prompt |
|---|---|---|
| "Write a Python function that checks if a number is prime." (300 tokens) | **15.3 tok/s** | 36 tok/s |
| "Explain how a hash map works in 5 sentences." (137 tokens) | **15.3 tok/s** | 36 tok/s |

**Answers:** correct and well formed (a documented `is_prime` with a square-root bound; an accurate hash map
explanation).

**What this shows:**
- **The engine loads and runs this architecture correctly.** The patches did not break the `qwen3next` path.
- **The CPU keeps up with memory.** Each token reads about 1.95 GB: 1,318 MiB of dense weights and 635 MiB of
  experts. At 15.3 tok/s that is about 30 GB/s, 67% of the measured RAM bandwidth.
- **The planner was pessimistic.** It estimated 11 tok/s, because it capped CPU throughput at 22 GB/s (Strata's
  figure for i-quants). It now uses 65% of RAM bandwidth and estimates 13.8 tok/s for this machine.

**Not exercised here:** the hot experts (no GPU) and the SSD tier (the model fits in RAM). Those need the target
PC.

**Implication for the target PC** (RTX 4070, 32 GB DDR4, about 40 GB/s), with the calibrated planner:

| Model | GPU + hot experts | Every expert on the CPU (`--cpu-moe`) | CPU only |
|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | ~74 tok/s | ~34 | ~13 |
| Qwen3.6-35B-A3B Q6_K | ~58 | ~11 | ~5 |
| Qwen3-Next-80B-A3B Q3_K_XL | ~34 | ~8 | ~5 |

The last two models do not fit in the PC's RAM. Without hot experts, part of every token comes from the SSD, which
is why the middle column drops so much.
