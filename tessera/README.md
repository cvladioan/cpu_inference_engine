# Tessera

**Run 30-120B mixture-of-experts models on a gaming PC:** hot experts on the GPU, the rest on the CPU, cold
ones from the SSD.

Tessera applies [Strata](https://github.com/Niko1221/Strata)'s architecture to any MoE model in GGUF format:
- the dense part and the most-used experts on the GPU;
- the other experts computed by the CPU at the same time;
- an SSD tier for experts that do not fit in RAM.

Its engine is [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) plus three patches (`engine/patches/`).
It serves an OpenAI-compatible API.

**Target PC:** RTX 4070 12 GB, Core i5-13400F, 32 GB DDR4, NVMe SSD, Windows 11 + WSL2. The plan, the model
choices and the estimates are in [docs/PLAN.md](docs/PLAN.md).

| Model | Estimate on the target PC |
|---|---|
| Qwen3.6-35B-A3B Q6_K (near-lossless) | ~50 tok/s |
| Qwen3-Next-80B-A3B Q3_K_XL (70B class) | ~30 tok/s; 45-55 with speculative decoding (roadmap) |
| any dense 70B | ~2 tok/s: bandwidth-bound, see the plan |

These are bandwidth-model estimates; `scripts/bench.sh` measures the real numbers.

**First real run** ([docs/RESULTS.md](docs/RESULTS.md)): Qwen3-Next-80B-A3B UD-Q3_K_XL on an 8-vCPU VM with no GPU
answers correctly at **15.3 tok/s**.

**Status (v0.1):**
- The engine change and the tooling are complete.
- Outputs are verified to match the plain engine in every configuration (CPU-only test, `tests/run_tests.sh`).
- The first GPU run, and so the real speed, is still ahead.

## Quick start (Windows 11 + WSL2)

**1. In PowerShell:** give WSL most of the RAM, and no swap. Write `%USERPROFILE%\.wslconfig`, then run
`wsl --shutdown`:

```ini
[wsl2]
memory=26GB
swap=0

[experimental]
autoMemoryReclaim=disabled
```

The NVIDIA driver stays the Windows one; do not install a driver inside WSL.

**2. In Ubuntu (WSL):**

```bash
git clone https://github.com/cvladioan/cpu_inference_engine.git ~/cpu_inference_engine
cd ~/cpu_inference_engine && git checkout claude/modest-goodall-ryj2y7 && cd tessera
cp tessera.env.example tessera.env        # edit MODEL_REPO / MODEL_PATTERN / THREADS if you like

scripts/setup-wsl.sh                      # build tools, CUDA 12.8 (NVIDIA's WSL repo), downloader (~10 min)
scripts/build.sh                          # the engine, for your GPU (20-60 min the first time)
tests/run_tests.sh                        # must end with PASS
scripts/download.sh --list                # the quant files of MODEL_REPO and their sizes
scripts/download.sh                       # e.g. Qwen3-Next-80B-A3B UD-Q3_K_XL, ~34 GB
scripts/calibrate.sh                      # the hot-expert profile for this model (5-20 min, once)
scripts/serve.sh --plan                   # what goes where, and the speed estimate
scripts/serve.sh                          # the server: http://127.0.0.1:8080/v1
```

**3. Measure:**

```bash
scripts/bench.sh                          # every expert on the CPU vs Tessera's tiers: tok/s, same answers
scripts/bench.sh --modes hot --threads 6,8,10
```

To use a model you already have, set `MODEL_FILE=/path/to/model.gguf` in `tessera.env` (the first shard of a split
model).

## CPU only

With no GPU, set `CPU_ONLY=1` in `tessera.env`, or leave it on `auto`, which switches on by itself when the engine
was built without CUDA or no NVIDIA GPU is visible. Then:
- `setup-wsl.sh` skips the GPU check and the CUDA toolkit;
- `build.sh` builds for the CPU (about 10 minutes);
- `calibrate.sh` has nothing to do (the profile only chooses what goes to VRAM);
- `serve.sh` keeps everything in RAM, and turns on the SSD tier when the model does not fit.

```bash
CPU_ONLY=1 scripts/setup-wsl.sh
CPU_ONLY=1 scripts/build.sh
tests/run_tests.sh
scripts/serve.sh --plan        # shows "CPU only", the RAM plan and the estimate
scripts/serve.sh
scripts/bench.sh --modes cpu --threads 6,8,10
```

Expect about a sixth of the GPU speed on this PC. The CPU reads the dense part and every expert from RAM at about
40 GB/s, where the GPU reads its share at 504 GB/s. Rough estimates for the target PC (`tools/plan.py --cpu-only`):

| Model | CPU only | With the GPU |
|---|---|---|
| gpt-oss-20b, Qwen3.6-35B-A3B Q4_K_M (fit in RAM) | ~10-15 tok/s | ~70 tok/s |
| Qwen3.6-35B-A3B Q6_K, Qwen3-Next-80B-A3B Q3_K_XL (larger than RAM: SSD tier) | ~5 tok/s | 30-50 tok/s |
| a dense 27-32B model at 4 bits | ~2 tok/s | ~2-4 tok/s |

For CPU-only servers with more memory channels (the original goal of this repository), use
[`../deploy/`](../deploy/README.md). It adds NUMA placement and a systemd service.

## Use it from a coding agent (pi)

Any client of the OpenAI API works. `serve.sh` passes `--jinja`, which tool calls need.

To use [pi](https://github.com/badlogic/pi-mono) from another machine:

1. **Reach the server through an SSH tunnel.** The server can then stay on `127.0.0.1` with no key:

   ```bash
   ssh -N -L 18080:127.0.0.1:8080 user@server
   ```

2. **Add the endpoint** to `~/.pi/agent/models.json` (on Windows, `%USERPROFILE%\.pi\agent\models.json`):

   ```json
   {
     "providers": {
       "tessera": {
         "baseUrl": "http://127.0.0.1:18080/v1",
         "api": "openai-completions",
         "apiKey": "none",
         "models": [
           { "id": "local", "name": "Tessera", "contextWindow": 32768, "maxTokens": 8192, "reasoning": false }
         ]
       }
     }
   }
   ```

   - `contextWindow` is `CTX`.
   - Set `"reasoning": true` for a thinking model.

3. **Choose the model** with `/model` in pi.

**Speed:**
- The first request processes pi's system prompt and tool definitions (a few thousand tokens) at prompt speed. On a CPU that takes up to a minute.
- Later turns reuse the server's prompt cache, so only the new messages are processed.
- ik_llama.cpp keeps context checkpoints by default (`--ctx-checkpoints 32`). Hybrid models such as Qwen3-Next need them to reuse that cache.

**Native Windows:**
- Run pi from PowerShell or Windows Terminal.
- MobaXterm's local shell turns npm's launcher path into `C:\drives\c\...`, so pi fails there with `Cannot find module`.
- pi runs its `bash` tool through Git Bash.

## How it works

```
 GPU  dense layers + KV cache + HOT EXPERTS (the profile's most used, copied at load)
  |        router picks 10 experts ─┬─ hot ones:   computed on the GPU ──┐
  |                                 └─ the others: computed on the CPU ──┴─ sum
 RAM  every expert, mapped from the model file
 SSD  experts that do not fit in RAM, read on demand (explicit expert cache)
```

- **Calibration** (`scripts/calibrate.sh`) runs the model on varied prompts and records which experts every layer
  routes to (`TESSERA_ROUTING_TRACE`). `tools/make_profile.py` ranks them and prints how concentrated the use is.
- **Planning** (`tools/plan.py`) reads the model's header and this PC's VRAM and RAM:
  - how many MiB of hot experts fit next to the dense layers, the KV cache and the buffers;
  - whether the SSD tier is needed;
  - a speed estimate.
- **Serving** (`scripts/serve.sh`) starts `llama-server` with `--cpu-moe`, `--hot-experts <profile>` and
  `--hot-experts-mib <budget>`, plus `--defer-experts --expert-cache <MiB>` when the experts do not fit in RAM.
- **In each MoE layer:**
  - the routed ids are split: the GPU's list holds indices into the hot copies, the CPU's holds the rest;
  - an id of -1 means "not mine";
  - the GPU half is queued first, so both halves run at the same time;
  - the two sums are added.

## Layout

| Path | What |
|---|---|
| `engine/patches/` | `0001` SSD expert cache, `0002` its memory guard, `0003` hot experts + routing trace |
| `scripts/` | `setup-wsl.sh`, `build.sh`, `download.sh`, `calibrate.sh`, `serve.sh`, `bench.sh`, `lib.sh` |
| `tools/` | `gguf_info.py` (what a model needs from each tier), `plan.py` (budgets and speed estimate), `make_profile.py`, `bench.py` |
| `tests/` | `run_tests.sh`: the whole flow on a tiny random MoE, CPU only |
| `prompts/calibration.txt` | varied prompts for the profile: edit to match your use |
| `docs/PLAN.md` | the plan: models, estimates, architecture, roadmap, risks |

## Engine options added by the patches

| Option | Meaning |
|---|---|
| `--hot-experts FILE` | profile: `layer expert` per line, most used first |
| `--hot-experts-mib N` | VRAM for the hot experts (needs the experts in host memory: `--cpu-moe` or `-ot exps=CPU`) |
| `--expert-cache N` (with `--defer-experts`) | SSD tier: an N MiB RAM cache for experts of a model larger than RAM (Linux/WSL) |
| `TESSERA_ROUTING_TRACE=file` | environment variable: append every MoE layer's routed ids to `file` |

## Credits

- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (MIT): the engine.
- [Strata](https://github.com/Niko1221/Strata) (MIT): the tier architecture, profiles and routing traces this
  project generalizes.
- The SSD expert cache and its tests come from
  [cpu_inference_engine](https://github.com/cvladioan/cpu_inference_engine).
