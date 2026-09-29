# Handoff: status and next steps

Last updated: 2026-09-29. Branch: `claude/modest-goodall-ryj2y7`.

Read this first when resuming on another machine. It records the goal,
everything built and measured so far, what was learned, and the exact plan for
the next piece of work.

## 1. The goal

Run big Mixture-of-Experts (MoE) models, DeepSeek-V4-Flash first, on **common
CPUs**: an ordinary desktop, with no GPU and no server hardware. This is meant
to be done by **researching and building new technology**, not only by
deploying existing engines.

The owner's reference machine:
- Windows (WSL2)
- 32 GB RAM
- 496 GB SSD
- Intel Core i5 13th gen: 6 P-cores, AVX2 + AVX-VNNI, no AVX-512

## 2. The core constraint

| Fact | Number |
|---|---|
| DeepSeek-V4-Flash size | 284B total parameters, 13B active per token |
| Model file (Unsloth GGUF) | ~155 GB at 4-bit, ~162 GB at Q8 |
| Bytes read per generated token | ~7 GB at 4-bit, ~9.6 GB at native precision |
| Desktop RAM bandwidth (2-channel DDR5) | ~60-90 GB/s |
| NVMe SSD read speed | ~3-7 GB/s |

What this means:

- CPU decoding is memory-bandwidth bound: every token must read every active
  weight.
- On the reference PC the model does not fit in RAM. Experts must come from
  the SSD, which gives about 1-2 tok/s.
- Even if everything fit in RAM, a 2-channel desktop tops out around 5 tok/s.
- A 12-channel server socket (about 460-845 GB/s) reaches 20-40 tok/s. That
  path is fully built (section 4), but it is not "common CPU".

So new technology for common CPUs has to either:
- **(a) read fewer bytes per token**, or
- **(b) serve those bytes from RAM instead of SSD** (higher expert cache hit
  rate).

The work in progress (section 7) targets (b), then (a).

## 3. Repository map

| Path | What it is |
|---|---|
| `docs/PLAN.md` | Research plan: models, hardware sizing, engine architecture, roadmap, risks. Section 12 covers the SSD memory tier. |
| `docs/HANDOFF.md` | This file |
| `docs/BOTTLENECKS.md` | **Measured analysis of what blocks big models on common CPUs**: blockers ranked, time budget per token, research agenda |
| `tools/roofline.py` | Bandwidth roofline calculator: decode tok/s per hardware and quant, batching, MTP, SSD streaming, `--target-tps` |
| `tools/engine_profile.py` | Runs the measurements on any machine: DRAM bandwidth, decode efficiency, prefill GFLOP/s, per-layer overhead |
| `tools/membw.c` | DRAM read-bandwidth benchmark per thread count |
| `tools/ram_limit.py` | Holds RAM to emulate a smaller-RAM PC (forces SSD streaming) |
| `docs/EXPERT_CACHE.md` | **Explicit expert cache**: design, correctness, results (4.4-6x over the page cache) |
| `engine/patches/` | Patches to ik_llama.cpp, applied by `deploy/build.sh` (currently: the explicit expert cache) |
| `deploy/test/greedy_outputs.py` | Greedy completions plus top-5 probabilities as JSON, and `--compare` to prove an engine change does not alter results |
| `tools/expert_pin.py` | Prototype: pins non-expert weights plus a budget of experts in the page cache with `mlock`; no engine changes |
| `deploy/` | Working deployment of DeepSeek-V4-Flash on ik_llama.cpp. Runbook: `deploy/README.md` |
| `deploy/check_host.sh` | Host readiness check: CPU ISA, RAM, DIMMs, NUMA, disk/NVMe, WSL `.wslconfig` |
| `deploy/build.sh` | Builds ik_llama.cpp at the pinned commit (`-march=native`) |
| `deploy/download_model.sh` | Downloads the model (split or single file) and an optional DSpark draft |
| `deploy/serve.sh` | Starts the server. Handles NUMA placement (none, interleave or per-node) and SSD expert streaming (`EXPERT_STREAMING`). Passes the API key through a pipe. |
| `deploy/tune.py` | Auto-tuner: tries threads, `-rtr`, speculative decoding (n-gram, DSpark) and `-ser`, measures real tok/s, checks `--target`, and `--apply` saves the winner |
| `deploy/quickstart.sh` | One command: check, build, download, tune, and optional service install |
| `deploy/smoke_test.py`, `deploy/bench.sh` | Measure the API (TTFT, tok/s, concurrency) and run llama-sweep-bench |
| `deploy/install_service.sh`, `deploy/systemd/`, `deploy/nginx.conf.example` | systemd service and optional load balancer |
| `deploy/test/make_tiny_gguf.py` | Writes random-weight GGUFs: tiny dense, or MoE of any size (`--size-gb`) for streaming tests |
| `deploy/test/run_local_test.sh` | End-to-end regression test: server, API key, smoke test, tuner, bench |

Engine: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), pinned to
`d741de5074cd424dd3ba7cfc4d9b7649f1eb0463` (2026-09-28).

## 4. What was built and verified

All of this was tested in a cloud VM (4 cores with AVX-512 VNNI, 15 GB RAM, no
GPU, Hugging Face blocked). The tests used the real ik_llama.cpp build with
random-weight test models; no real model weights were available.

- **Build** at the pinned commit works. The AVX-512 VNNI kernels are present
  (16,219 `vpdpbusd` instructions in `libggml.so`).
- **Server:** `run_local_test.sh` passes. It covers streaming chat, 2
  concurrent requests, API-key enforcement (401 without a key), the tuner and
  the benchmark. `shellcheck` is clean.
- **NUMA modes** `none`, `interleave` and `per-node` all start. The per-node
  supervisor was checked with a faked 2-node topology: killing one instance
  stops the rest, and SIGTERM stops all instances cleanly with exit code 0.
- **SSD expert streaming** was tested with a 20 GB MoE model on 15 GB of RAM.
  The server was ready in 4 s: 0.4 GB of dense weights loaded, 18.75 GB of
  experts deferred to disk.
  - First request (cold): 9.7 s to the first token, 5.7 tok/s.
  - Second request (warm): 0.14 s to the first token, 16.4 tok/s.
  - Prompt batch size matters: 1024-token batches process prompts 5.5x faster
    than 128-token ones (95 vs 17 tok/s). Streaming mode therefore uses 2048.
- **Tuner:** full staged search on a 2.4 GB MoE test model; `--apply` writes
  `config.env` with a backup. The quickstart also ran end to end.
- **Draft model wiring:** `-md` is passed, speculative decoding runs, and a
  clear error appears when a draft is missing.

**Not yet verified:** any real model; real Windows/WSL2; a real 12-channel
server.

## 5. Research findings

Sources are in `docs/PLAN.md` section 13; new ones are listed below.

**Engine status**
- ik_llama.cpp is the fastest CPU engine for DeepSeek-V4 (PR #2165).
  Measured CPU-only on a Threadripper 3995WX (8-channel DDR4): 9.7 tok/s
  generation and 107 tok/s prompt processing with Q4_K_M. The roofline model
  is calibrated on this result (engine efficiency about 0.45).
- ik_llama.cpp already has SSD streaming: `--defer-experts` and
  `--prefetch-experts` (`ggml/src/ggml-moe-prefetch.{h,cpp}`). The prefetch is
  reactive, meaning it starts only after the current layer has routed. Cache
  replacement is left to the kernel's page cache (LRU-like), and prompt
  sweeps are marked `MADV_COLD`.

**Speculative decoding for V4-Flash**
- The 0731 checkpoint ships a DSpark draft, not MTP.
- DSpark: +80% with everything on the GPU (DGX Spark), but only +10-15% when
  the experts run on CPU (llama.cpp PR #25784). ik PR #2280 measured −12.9%
  overall on its hybrid setup.
- MTP: 81% acceptance at `n_max=1`, but no throughput gain with CPU experts
  (ik PR #2309).
- Conclusion: speculative decoding is a small lever on CPU, because verifying
  k tokens touches up to k×6 experts.

**Prior art for expert caching** (to cite and build on)
- **ReMoE** (ICML 2026, <https://github.com/BUAA-OSCAR/ReMoE>): fine-tunes
  only the router so adjacent tokens reuse experts, which raises cache hit
  rate. It needs training per model. There is a DeepSeek-V2-Lite-Chat-ReMoE
  GGUF on Hugging Face (`Zhu149248/DeepSeek-V2-Lite-Chat-ReMoE-GGUF`), a
  useful baseline to compare against.
- **HOBBIT**: runs cache-missed, low-importance experts at low precision
  instead of waiting for the full-precision copy.
- **Cache-conditional experts** (Qualcomm AI Research, mobile): biases routing
  toward experts already in cache, at inference time. Details not yet
  verified; read the paper before citing numbers.
- **Pre-gated MoE, ProMoE, AdapMoE, MoE-Infinity**: predict the next layer's
  experts and prefetch them.

**Hardware (only relevant for the server path)**
- EPYC 9004 Genoa: 12-channel DDR5-4800, 460 GB/s, for example Hetzner
  AX162-R (order 12 DIMMs) or AWS `r7a.48xlarge`. Estimated 23-38 tok/s at
  4-bit.
- Azure HBv5 has HBM3 at about 6.9 TB/s per node.

## 6. Decisions made

- Use ik_llama.cpp as the base engine and extend it, rather than writing an
  engine from scratch now.
- Default quant UD-Q8_K_XL (routed experts stay in native MXFP4). Use
  UD-Q4_K_XL for speed or small machines.
- Speculative decoding and `-ser` are off by default. The tuner measures them
  per machine.
- The model is kept memory-mapped (never `--mlock`) when streaming experts.
  Prompt batches are 2048 tokens when streaming.

## 7. Bottleneck analysis (done 2026-09-29) and the next step

The full write-up is in `docs/BOTTLENECKS.md`; every number there was
measured in this session. Key findings:

- **Page-cache cliff.** With free RAM close to the model size, SSD-streamed
  decode fell from 13.9 to 0.4 tok/s. The engine read 2.75 GB per token,
  twice all the weights a token uses.
  - Cause: least-recently-used eviction under the fixed layer order, plus
    read-ahead. The prefetcher is not the cause: plain `mmap` behaves the
    same.
  - **This is the largest software-fixable loss.**
- **Pinning prototype** (`tools/expert_pin.py`): non-expert weights plus 30%
  of experts **doubled** tok/s (0.4 to 0.8) and cut reads 45%. Over-pinning
  hurts once the cache has room (1.3 to 1.1 tok/s at 4.6 GB free), so the
  real fix is an adaptive cache.
- **Engine efficiency:** 49-54% of DRAM bandwidth at 4-bit, 59-68% at 8-bit.
  4-bit is compute-bound per core. `-rtr` adds 11%.
- **Per-layer overhead:** 28-36 µs on 1 thread, 43-45 µs on 2-4 threads.
  Threads hurt on tiny layers.
- **Prefill:** 470-520 GFLOP/s on 4 cores, about 15% of int8 peak. That means
  about 18-25 tok/s prompt speed for V4-Flash on a desktop.
- **Bandwidth:** about 9 GB/s per core, scaling linearly to 34.6 GB/s on 4
  cores.

**Revised order of work:**
1. **R1: explicit expert cache. DONE (2026-09-29).** Full write-up in
   `docs/EXPERT_CACHE.md`; the patch is in
   `engine/patches/0001-explicit-expert-cache.patch` and `deploy/build.sh`
   applies it.
   - Results against the page cache, same VM and model: 3.6 GB free
     0.5 → 2.2 tok/s; 4.0 GB free 0.6 → 3.9 tok/s; 4.6 GB free 1.9 →
     8.3 tok/s. Disk reads fell 5-8x.
   - Outputs are bit-identical; the regression test checks this.
   - Engine code lives in `src/llama-expert-cache.{h,cpp}` of ik_llama.cpp,
     hooked in `llm_build_moe_ffn`.
   - Deploy: `EXPERT_CACHE_MIB=auto` (default), `EXPERT_CACHE_HEADROOM_MIB`.

   The original design goals were:
   - non-expert weights always resident
   - a frequency-aware, scan-resistant pool of whole experts
   - exact-slice reads, no read-ahead

   Test bed: the same VM, with `tools/ram_limit.py` and the V4-shaped Q8_0
   model. Target: at least 3x tok/s at the tight memory level.
2. **R2: Cache-Aware Routing** (below) on top of R1.
3. **R3:** predictive prefetch.
4. **R4:** fewer bytes per token.

Test models are not stored in the repo. Regenerate them with:

```bash
python3 deploy/test/make_tiny_gguf.py --embd 4096 --ff 2048 --experts 16 --used 4 --layers 8 --heads 32 f32.gguf
llama-quantize --pure f32.gguf v4like-Q8_0.gguf Q8_0      # 3.8 GB; each token uses 0.53 GB non-expert + 0.86 GB experts
```

## 7b. Planned: Cache-Aware Routing (CAR)

**Idea (training-free, runtime-only, works on any MoE GGUF):**
- On a common PC, most experts live on the SSD. Each cache miss costs a slow
  SSD read.
- MoE routers often have near-ties among the top experts.
- CAR adds a small bonus **β** to the routing scores of experts that are
  already in RAM. This is used only for *selecting* experts; the gate weights
  stay unbiased, the same way DeepSeek-V3 uses `exp_probs_b`.
- Near-ties then go to cached experts, which raises the hit rate and cuts SSD
  reads.
- Optional **miss-skip τ**: drop a selected expert that is not in RAM when its
  normalized gate weight is below τ, then renormalize the remaining weights.

Differences from prior art:
- Unlike ReMoE, no fine-tuning, so it works immediately for DeepSeek-V4-Flash.
- It is implemented in the CPU engine, with the SSD and page cache as the
  memory hierarchy.
- It composes with ReMoE-style routers and with prefetching.

**Where to implement it in ik_llama.cpp** (pinned commit):
- **Routing:** `src/llama-build-context.cpp`, function `llm_build_moe_ffn`
  (around line 1459).
  - `selection_probs = probs` (+ `exp_probs_b`) is at about line 1527.
  - `ggml_top_k(ctx, selection_probs, n_expert_used)` is at about line 1549.
  - The callback names are `ffn_moe_logits`, `ffn_moe_probs` and
    `ffn_moe_topk`.
- **DeepSeek-V4 graph:** `src/graphs/build_deepseek4.cpp` calls
  `llm_build_moe_ffn` at about lines 1660, 1923 and 2668.
  - V4 uses hash routing in its first 3 MoE layers; CAR does not apply there.
- **Custom op for recording selections:** `ggml_map_custom1` is available
  (`ggml/include/ggml.h` line 2819).
- **Existing expert reduction:** `-ser Kmin,thresh` is parsed in
  `common/common.cpp` around line 1999, as a pair (Kmin, threshold).
  `ggml_top_k_thresh` is commented out next to the top-k call.
- **Expert byte ranges:** `src/llama-model-loader.cpp`,
  `build_expert_tensor_index()`.

**Implementation plan (prototype):**
1. **Per-layer residency tracker** (host memory): an LRU set of expert IDs
   with capacity C per layer.
   - Default C = (RAM available for experts) / (bytes per expert per layer).
   - Optional later: probe real page-cache residency with `mincore()` on
     expert ranges.
2. **Record selections:** after `ggml_top_k`, add
   `ggml_map_custom1(selected_experts, car_record)` and expand it into the
   graph. Its callback pushes the chosen IDs into the tracker and counts hits
   and misses.
3. **Bias tensor:** a per-layer `car_bias` tensor of size `[n_expert]` in a CPU
   buffer. `selection_probs = probs + car_bias`. After each decode step, write
   β for resident experts and 0 for the rest with `ggml_backend_tensor_set`.
   This keeps graph reuse valid.
4. **Miss-skip (τ):** a masked renormalization of `weights` for non-resident
   experts below τ. It can be a second custom op, or done in the recording
   callback by editing the weights.
5. **Flags:** `--car-bias β`, `--car-capacity C`, `--car-skip τ`,
   `--car-trace file` (dumps per-token routing for offline simulation).
6. **Stats:** hit rate per layer, and expert bytes fetched per token, logged
   periodically.
7. **Delivery:** keep the change as a patch in this repo (for example
   `engine/patches/0001-cache-aware-routing.patch`), applied by
   `deploy/build.sh`. Upstream it later.

**Experiment plan:**
- **Models** (need Hugging Face access; the owner's PC has it):
  - DeepSeek-V2-Lite-Chat GGUF (16B, 64 experts, top-6; DeepSeek routing; fits
    in 15 GB)
  - OLMoE-1B-7B (64 experts, top-8)
  - Qwen3-30B-A3B or Qwen3.5/3.6-35B-A3B (a 256-expert design)
  - DeepSeek-V4-Flash itself on the PC
  - The ReMoE GGUF, as the fine-tuned-router baseline
- **Emulate a small-RAM PC:** limit the page cache available to the model,
  either by locking a memory-hog buffer or with a cgroup/WSL memory limit.
  This forces streaming.
- **Metrics:**
  - Expert hit rate.
  - SSD bytes per token, from `/proc/<pid>/io` `read_bytes` or `vmstat`
    major faults. This number is hardware-independent.
  - Generation tok/s.
  - Quality: `llama-perplexity --kl-divergence` against baseline logits
    (mean KL, top-1 agreement), plus perplexity on wikitext and a code sample.
- **Sweep** β ∈ {0, 0.01, 0.02, 0.05, 0.1} (relative to prob scale), τ ∈
  {0, 0.05, 0.1}, and cache capacity from 10% to 60% of experts.
- **Success criterion:** at least 2x fewer SSD bytes per token at under 1%
  top-1 disagreement.

**Next techniques after CAR** (in order):
1. **Predictive prefetch.** Apply layer l+1's router to layer l's hidden state
   so I/O overlaps with compute. Worth up to about 2x when SSD-bound.
2. **Expert-aware cache policy.** Frequency-based pinning via `mlock` of hot
   expert ranges, so prompt sweeps cannot evict the decode working set.
3. **Tiered precision.** Low-bit copies of cold experts stay RAM-resident and
   are used on a miss (HOBBIT-like), with full precision on the SSD.
4. **Domain-profiled expert pruning** (REAP-style) to fit a smaller RAM for a
   known workload.

## 8. Blockers

- **The cloud environment blocks Hugging Face** (`huggingface.co`, `*.hf.co`),
  Ollama, ModelScope and other model hosts. Fix it in the environment
  settings: session title bar → cloud environment menu → Edit → Network
  access. Allow those domains or choose a broader access level. Or run the
  experiments locally, where downloads work.

## 9. How to resume on another machine

```bash
git clone -b claude/modest-goodall-ryj2y7 https://github.com/cvladioan/cpu_inference_engine.git
cd cpu_inference_engine
cat docs/HANDOFF.md                      # this file
python3 tools/roofline.py                # all estimates
```

To get the engine and test harness running (Linux or WSL2; see
`deploy/README.md`, "Test on a desktop PC"):

```bash
cp deploy/config.env.example deploy/config.env    # set INSTALL_DIR etc.
deploy/check_host.sh
deploy/build.sh
deploy/test/run_local_test.sh                     # ~1 minute, no downloads
```

To start the CAR work (section 7):

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp
git checkout d741de5074cd424dd3ba7cfc4d9b7649f1eb0463
# edit src/llama-build-context.cpp (llm_build_moe_ffn), then add flags in common/common.cpp
```

If you continue with Claude Code, point it at this file:

> Read docs/HANDOFF.md, docs/BOTTLENECKS.md and docs/EXPERT_CACHE.md, then continue with R2 (cache-aware routing) from section 7b, built on the expert cache.
