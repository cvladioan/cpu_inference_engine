# Explicit expert cache

An engine change for running Mixture-of-Experts models that are larger than
RAM on ordinary CPUs:
- It replaces the OS page cache for expert weights with a cache the engine
  manages.
- It is delivered as a patch to ik_llama.cpp:
  [`engine/patches/0001-explicit-expert-cache.patch`](../engine/patches/0001-explicit-expert-cache.patch).
  `deploy/build.sh` applies it automatically.
- It is enabled with `--expert-cache <MiB>`.

**Result on the test machine:** **4.4-6x faster decoding and 5-8x less disk
traffic** than ik_llama.cpp's page-cache streaming, at every memory level
measured. Outputs are bit-identical.

## 1. The problem it solves

When a MoE model does not fit in RAM, llama.cpp-style engines memory-map the
file and let the OS page cache decide what stays in memory.
`docs/BOTTLENECKS.md` (section 2.6) measured what goes wrong:

- **The layer loop defeats the OS's eviction.** Each token walks layers
  0..N in the same order. Once the working set is a little larger than free
  RAM, least-recently-used eviction throws out exactly the pages needed next,
  and the hit rate collapses.
- **Read-ahead amplifies traffic.** Read-ahead, and prefetched pages evicted
  before use, multiply disk reads. We measured 2.7 GB read per token when a
  token needs 1.4 GB.
- **Non-expert weights get evicted too.** Attention and shared-expert weights,
  which every token reads, are evicted and re-read.

## 2. How it works

All changes are in `src/llama-expert-cache.{h,cpp}`, plus about 30 lines of
wiring.

- **Engine-owned memory at the same addresses.**
  - When the graph for a MoE layer is first built, the page-aligned interior
    of every expert tensor (`ffn_{up,gate,down,gate_up}_exps`) is replaced
    with anonymous memory at the same virtual addresses (`mmap MAP_FIXED`).
  - Kernels keep their pointers; no kernel changes.
  - The OS can no longer evict or read ahead there.
- **Load exactly what the layer needs, just before it runs.** Two small graph
  operations are inserted between routing and the expert matmuls:
  - **plan** (one thread): reads the chosen expert ids, counts hits, evicts by
    policy, and lists the exact byte ranges to load.
  - **load** (all compute threads): reads those ranges in parallel with
    `pread` and `O_DIRECT`, so there is no read-ahead and no second copy in
    the page cache.
  - The CPU backend puts a barrier after every graph operation, so the experts
    are resident before any matmul reads them.
- **Non-expert weights are resident.**
  - On first use, every other byte range of the model file (attention, shared
    experts, embeddings, output head) becomes anonymous memory read once.
  - Any expert tensor not managed by the cache counts as non-expert and is
    fully loaded, so nothing can ever read unloaded memory.
- **Eviction policy: segmented LRU over whole experts.**
  - An expert hit twice during decoding moves to a protected segment (up to
    80% of the budget).
  - Experts loaded while processing a prompt enter at the evict-first end, so
    one prompt sweep over all experts cannot flush the hot set.
  - When the budget is smaller than one token's experts across all layers,
    misses are admitted evict-first as well. The resident set then stays
    stable instead of cycling (the standard remedy for loop access larger
    than the cache).
  - Measured at a 512 MB budget: hit rate went from 2.3% to 17.9%. That is
    close to the best possible for uniformly random routing, which is the
    fraction of experts that fit (about 16%).
- **One cache per model.** The cache owns the model's expert memory, so every
  context on the model shares it, even one created without the flag.

## 3. Correctness

- Greedy decoding, with top-5 token probabilities for 3 prompts × 48 tokens,
  is **bit-identical** to the unpatched engine in every tested case:

  | Model | Budget | Evictions | Hit rate |
  |---|---|---|---|
  | 3.8 GB V4-shaped model | 8 GiB | 0 | - |
  | 3.8 GB V4-shaped model | 1 GiB | 424 | - |
  | 3.8 GB V4-shaped model | 512 MiB | 4,721 | 2% |
  | small MoE in `deploy/test/run_local_test.sh` | 4 MiB | 2,349 | - |

- The regression test runs this comparison automatically whenever the build
  has the patch (`deploy/test/greedy_outputs.py --compare`).

## 4. Results

Test setup:
- The same VM as `docs/BOTTLENECKS.md`: 4 cores, 15 GB RAM, disk at about
  1 GB/s.
- Synthetic model with DeepSeek-V4-Flash-sized experts: 8 layers × 16 experts,
  4 active per token, Q8_0, 3.8 GB. Each token uses 0.53 GB of non-expert
  weights plus 0.86 GB of experts.
- Free RAM squeezed with `tools/ram_limit.py`.
- 48 sampled tokens after warm-up; disk reads from `/proc/vmstat`.
- Page cache and expert cache measured back to back in the same run.

| Free RAM | ik_llama.cpp page cache (`--defer-experts --prefetch-experts`) | Expert cache | Speed-up | Disk reads |
|---|---|---|---|---|
| 3.6 GB | 0.5 tok/s, 2,700 MB/token | **2.2 tok/s**, 527 MB/token (1.0 GiB budget, 34% hit rate) | **4.4x** | 5.1x less |
| 4.0 GB | 0.6-0.7 tok/s, 1,815-1,851 MB/token | **3.9 tok/s**, 259 MB/token (auto budget 1.38 GiB, 63% hit rate) | **~6x** | ~7x less |
| 4.6 GB | 1.9 tok/s, 580 MB/token | **8.3 tok/s**, 76 MB/token (2.0 GiB budget, 84% hit rate) | **4.4x** | 7.6x less |
| (all in RAM) | 13.9 tok/s | | | |

- Page-cache baselines vary between runs at the same level (for example 1.3,
  1.6 and 1.9 tok/s at 4.6 GB), so each comparison uses the baseline from its
  own run.
- These are uniformly random routings from random weights, the worst case for
  any cache. Real models route with skew and locality, so hit rates should be
  higher.

## 5. What it means for DeepSeek-V4-Flash on a 32 GB PC

Estimate, not yet measured on the real model:
- **Budget:** automatic sizing gives about 18 GB of cache next to about 4 GB
  of non-expert weights (UD-Q4_K_XL).
- **Hit rate:** that is about 12% of all expert bytes. Uniform routing would
  give about a 12% hit rate; real skew should give more.
- **Per token:** the SSD then serves about 3.2 GB (all of it useful, none
  amplified). At about 5 GB/s that takes about 0.64 s, plus about 0.1 s from
  RAM.
- **Speed:** about **1.3 tok/s**, against about 0.3-0.5 tok/s with the page
  cache (docs/BOTTLENECKS.md section 3).

With waste removed, the SSD's bandwidth becomes the limit. The next steps
attack exactly that:
- **Cache-aware routing** raises the hit rate.
- **Predictive prefetch** overlaps SSD reads with compute.
- **Lower-bit experts** reduce the bytes per miss.

## 6. Using it

With `deploy/`:
- `EXPERT_CACHE_MIB=auto` (the default) switches the cache on whenever experts
  stream from SSD (`EXPERT_STREAMING`).
- Auto sizes it as 90% of (free RAM - non-expert weights -
  `EXPERT_CACHE_HEADROOM_MIB` - `CACHE_RAM_MIB`).
- `deploy/serve.sh --plan` shows the chosen budget.
- If the binary was built without the patch, the scripts fall back to the page
  cache and say so. Rerun `deploy/build.sh` to apply the patch.

Directly:

```bash
llama-server -m model.gguf --defer-experts --expert-cache 18000 ...
LLAMA_EXPERT_CACHE_LOG=2000 llama-server ...   # log hit rate every 2000 layer calls; always logged at exit
```

**Sizing:** the cache's memory cannot be reclaimed by the OS. A budget that
leaves too little free RAM gets the process (or another one) killed by the
out-of-memory killer, so size conservatively. Auto sizing does.

## 7. Limitations and next steps

**Limitations:**
- **Linux only.** It needs `mmap` of the model file, so it cannot be used with
  `--no-mmap`, `-rtr` repacking, or huge-page anonymous mappings. Tensors that
  are not file-mapped are simply not managed.
- **Contexts must decode one at a time.** All contexts on a model share one
  cache, which is not thread-safe across contexts decoding in parallel.
  `llama-server` decodes sequentially.
- **Replaces `--prefetch-experts`,** which is disabled automatically.
- **Validated on synthetic weights and on ik_llama.cpp `d741de5`.** Real-model
  validation (DeepSeek-V2-Lite, OLMoE, V4-Flash) is pending, because Hugging
  Face is blocked in the development environment.

**Next steps:**
- Cache-aware routing (docs/HANDOFF.md section 7b).
- Predictive prefetch of the next layers' experts.
- Tiered-precision experts.
- Upstreaming to ik_llama.cpp.
