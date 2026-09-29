# Tessera: plan

Tessera runs mixture-of-experts language models of 30-120B parameters on a gaming PC at reading speed or better.
It takes the architecture of [Strata](https://github.com/Niko1221/Strata), which runs one 125B model on a
12 GB GPU at 50-90 tok/s, and makes it work for any MoE model in GGUF format. Its engine is
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), plus Tessera's patches.

## 1. The target PC

| Part | The PC | What it limits |
|---|---|---|
| GPU | RTX 4070, 12 GB, 504 GB/s | how many experts fit in VRAM; the dense part's speed |
| CPU | i5-13400F: 6 performance + 4 efficiency cores, AVX2 + AVX-VNNI | computing the experts that are not in VRAM |
| RAM | 32 GB DDR4-3200, about 40 GB/s; about 26 GB for WSL2 | how many experts are served without the SSD |
| SSD | PCIe 3.0 NVMe, 1.5 GB/s sequential, 1.0 GB/s for 1 MB random reads | experts that fit neither in VRAM nor in RAM |
| OS | Windows 11 with WSL2 | 6 GB of RAM stay with Windows |

## 2. Why not a dense 70B model

A dense model reads all its weights for every token.

- A 70B model at 3 bits is about 27 GB. On this PC, 10 GB of it fits on the GPU and 17 GB stays in RAM.
- The CPU then reads 17 GB per token at about 35 GB/s: **about 0.5 seconds per token, 2 tok/s**.
- No engine changes that: it is the RAM's bandwidth.

A mixture-of-experts model splits the bulk of its weights into many small experts and uses only a few per token.

- Qwen3-Next-80B-A3B has 80B parameters, but only about 3B are used per token. At 3.5 bits that is about 0.65 GB
  of experts per token, instead of 27 GB.
- That is the model class where 40-50 tok/s is possible on this PC.

## 3. The models

Estimates from `tools/plan.py` (a bandwidth model of this PC, no speculative decoding). They compare options;
the measurement on the PC decides (`scripts/bench.sh`).

| Model | Size | Hot experts in VRAM | VRAM hit (est.) | Estimate |
|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 20 GB | 44% of experts | 75% | ~70 tok/s |
| **Qwen3.6-35B-A3B Q6_K** (near lossless) | 27 GB | 30% | 65% | **~50 tok/s** |
| Qwen3.6-35B-A3B Q8_0 | 35 GB | 21% | 58% | ~17 tok/s: the SSD tier starts to dominate |
| **Qwen3-Next-80B-A3B Q3_K_XL** (the 70B class) | 34 GB | 24% | 60% | **~30 tok/s**, ~45-55 with speculation (phase 4) |
| Qwen3-Next-80B-A3B Q4_K_XL | 44 GB | 17% | 54% | ~11 tok/s on 32 GB; ~30 on 64 GB |
| gpt-oss-120b MXFP4 | 60 GB | 10% | 45% | ~3 tok/s: large experts, too much per token |
| Llama-3.3-70B (dense) IQ3_XXS | 27 GB | - | - | ~2 tok/s (section 2) |

How the estimate works:
- It takes the time per token as the larger of the GPU part and the CPU part (they run at the same time), plus a
  fixed cost per layer for the hand-offs between GPU and CPU.
- It guesses the VRAM hit rate from the share of experts that fit, using Strata's measurement: 13% of the experts
  served about 50% of the reads with a static profile.
- With `--hit` it uses a measured rate instead.

**Quality.** The model you run today with Strata (Qwen3.8-Flash-Next Coder, 125B) is newer and larger than any
model in this table.
- Tessera's value is the models Strata does not run (any GGUF MoE, including future ones) at quants of your choice.
- Qwen3.6-35B-A3B at Q6_K gives near-full precision at about 50 tok/s.

## 4. How it works

```
          GPU (12 GB)                          CPU (RAM)                        SSD
 ┌─────────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────┐
 │ attention / DeltaNet, norms  │   │ all routed experts, mapped   │   │ the model file:       │
 │ shared experts, output head  │   │ from the file; the CPU       │   │ experts that do not   │
 │ KV cache                     │   │ computes the ones that are   │◄──│ fit in RAM, read on   │
 │ HOT EXPERTS: the most used,  │   │ not hot                      │   │ demand (expert cache) │
 │ copied at load (profile)     │   │                              │   │                      │
 └─────────────────────────────┘   └──────────────────────────────┘   └──────────────────────┘
```

For each MoE layer:
1. The router runs on the GPU and picks the experts.
2. On the CPU, two small operations turn the picked ids into two lists:
   - one for the GPU: indices into the hot copies, and -1 for the others;
   - one for the CPU: the original ids, and -1 for the hot ones.
   The CPU list's operation also takes the layer's activations and router weights as inputs, so they reach the
   host in the same step.
3. The GPU half is queued first, then the CPU half runs. The GPU computes the hot experts while the CPU computes
   the rest.
4. The two partial sums are added.

Both halves use the engine's existing kernels. They already skip id -1, because that is how the engine's `-ser`
option drops experts, on the CPU and on CUDA.

**Where the pieces come from:**

| Piece | Origin | In Tessera |
|---|---|---|
| Dense part and KV on the GPU, experts in RAM | llama.cpp / ik_llama.cpp (`--cpu-moe`) | used as is |
| Hot experts in VRAM, chosen by a routing profile | Strata's VRAM expert cache | `0003-hot-experts.patch`: for any GGUF MoE; the CPU and GPU halves run at the same time |
| Routing traces, profile builder | Strata's `--dump-routing`, `make_profile.py` | `TESSERA_ROUTING_TRACE`, `tools/make_profile.py`, `scripts/calibrate.sh` |
| SSD tier for experts that do not fit in RAM | cpu_inference_engine | `0001-explicit-expert-cache.patch`: direct-I/O reads, scan-resistant eviction |
| Memory guard | cpu_inference_engine | `0002`: the cache shrinks to what memory allows |
| Memory planner, speed model | new | `tools/plan.py` |

## 5. Status of v0.1

**Built:**
- the engine patch for hot experts;
- the routing trace;
- the planner, calibration, serving and benchmark scripts;
- the WSL2 setup (CUDA from NVIDIA's WSL repository);
- a regression test.

**Verified (CPU-only build, tiny random MoE, `tests/run_tests.sh`):**
- Every configuration answers like the plain engine: hot experts on, every expert hot, hot experts plus the SSD
  tier, and the SSD tier alone.
- Differences are at most 1.3e-4 in probability, from rounding (the split changes the order of a sum).
- A deliberately wrong expert mapping is caught: different text, probabilities off by up to 0.999.

**Not verified yet:**
- **Speed, and the concurrency of the GPU and CPU halves.** This needs a GPU; the first run on the target PC will
  show it.
- **The CUDA build on your PC.** The full engine builds without errors here against CUDA 12.0 for sm_89 (RTX 40),
  and its `llama-server` has `--hot-experts`. setup-wsl.sh installs 12.8, which is not tested here.
- **Architectures other than llama-style MoE.** Qwen3-Next and Qwen3.5/3.6 use the same MoE builder in the engine,
  so they should work, but they are untested.

## 6. Roadmap

| Phase | What | Expected gain | How we know |
|---|---|---|---|
| 1 (done) | Static hot experts, SSD tier, calibration, planner | est. 2.5x over `--cpu-moe` when the experts fit in RAM (35B Q4_K_M: 30 -> 74 tok/s), 4-5x when they do not (the hot experts also keep the CPU half off the SSD) | `bench.sh`: cpu vs hot |
| 2 | First GPU run: fix what it shows, tune threads, ubatch and hot budget | reach the estimates | `bench.sh --threads 6,8,10` |
| 3 | Adaptive hot set: every few tokens, swap in the experts the conversation uses most (Strata: VRAM hit 0.50 -> 0.72 on a 12 GB card) | +20-40% | hit rate from traces, tok/s |
| 4 | Speculative decoding: the model's MTP layer (Qwen3-Next) or n-gram drafts; one pass checks 2-4 tokens | x1.5-1.8 | tok/s at equal output |
| 5 | CPU kernels for this CPU: AVX-VNNI, a bigger share of the work for performance cores than efficiency cores | +10-25% of the CPU half | per-layer timings |
| 6 | Native Windows build: the RAM WSL keeps for itself goes to experts | +6 GB of RAM tier | |

## 7. Risks

- **The GPU/CPU hand-off per layer costs more than estimated.** CUDA graphs cannot span the CPU steps.
  - The estimate assumes 0.12 ms per layer.
  - If it is 0.3 ms, a 48-layer model loses about 9 ms per token.
  - Mitigations: fewer hand-offs (run the remap on the GPU) and speculation (phase 4), which pays each hand-off once
    per 2-4 tokens.
- **Hit rates are lower than Strata's.** Strata's model has 24,576 small experts; the skew differs per model. The
  calibration shows the real curve (`make_profile.py` prints how much of the traffic the top 5-50% of experts
  carry).
- **WSL2.** GPU access goes through the Windows driver, and WSL keeps 6 GB of RAM for itself (phase 6).
