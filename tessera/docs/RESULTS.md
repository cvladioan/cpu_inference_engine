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

### Tuning: threads, batch size, repacking

Measured with `llama-bench -p 1024 -n 32 -r 2` on the same VM and model:

| Threads | `-ub` | `-rtr` | Prompt (pp1024) | Output (tg32) |
|---|---|---|---|---|
| 6 | 512 | - | 67.7 tok/s | 17.8 tok/s |
| **8** | 512 | - | **85.5** | **21.1** |
| 6 | 2048 | - | 67.9 | 17.9 |
| 8 | 2048 | - | 86.5 | 21.0 |
| 8 | 2048 | 1 | 86.2 | 21.8 |

**What this shows:**
- **Use every vCPU on a VM.** Going from 6 threads to all 8 speeds up prompt processing by 26% and output by 18%.
- **At 21 tok/s, output uses 91% of the measured RAM bandwidth** (1.95 GB per token). The planner assumes 65%, which
  matches 6 threads. With all cores it underestimates.
- **A larger micro-batch does nothing here.** Prompt processing is limited by compute: about 3B active parameters,
  with the Q3_K dequantization, on 8 AVX2 cores.
- **Run-time repacking (`-rtr`) gains 3% on output and nothing on prompts.** It also turns off mmap and slows loading,
  so it is not worth using on this model.

### With a coding agent (pi)

[pi](https://github.com/badlogic/pi-mono) ran on a Windows workstation against this server, with 6 threads, in the
`models.json` setup from the README. Tool calls work through `--jinja`.

- **First request:** pi's system prompt and tool definitions are 1,423 tokens. They took 22 s (64 tok/s).
- **Later turns reuse the prompt cache**, so each processes only its new tokens: 16-76 tokens, at 30-46 tok/s (small
  batches). The hybrid model can reuse the cache because the engine keeps context checkpoints by default (up to 32 per
  slot, about 75 MiB each).
- **Output:** 16.5-17.4 tok/s at a 1.4K-3.3K context.
- **The cost of agent work is reading.** Every file and command output goes through prompt processing, so a 300-line
  file takes about a minute at 6 threads.

**Implication for the target PC** (RTX 4070, 32 GB DDR4, about 40 GB/s), with the calibrated planner:

| Model | GPU + hot experts | Every expert on the CPU (`--cpu-moe`) | CPU only |
|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | ~74 tok/s | ~34 | ~13 |
| Qwen3.6-35B-A3B Q6_K | ~58 | ~11 | ~5 |
| Qwen3-Next-80B-A3B Q3_K_XL | ~34 | ~8 | ~5 |

The last two models do not fit in the PC's RAM. Without hot experts, part of every token comes from the SSD, which
is why the middle column drops so much.
