# What blocks big models on common CPUs

Measured analysis, 2026-09-29. It answers: what stops a frontier MoE model
(DeepSeek-V4-Flash, 284B total / 13B active parameters) from running well on
an ordinary CPU, how much each blocker costs, and which ones software can fix.

Measurements come from ik_llama.cpp (commit `d741de5`) on a cloud VM:
- 4 cores of an Intel Xeon at 2.8 GHz, with AVX-512 VNNI
- 15 GB RAM, a virtual disk, no GPU

Real model weights were not reachable (Hugging Face is blocked), so the tests
use synthetic MoE models with DeepSeek-V4-Flash's per-expert shape: hidden
size 4096, expert width 2048. Their weights are random. That does not matter
for speed, bandwidth or I/O, but it does matter for routing patterns: section
6 marks what needs re-measuring with real weights.

Reproduce everything with `python3 tools/engine_profile.py` (sections 2.1-2.5)
and the commands in section 7.

## 1. Summary

1. **Bytes per token are the root constraint.** Decode reads every active
   weight once per token. DeepSeek-V4-Flash at 4-bit reads about 7 GB per
   token:
   - about 3.4 GB of non-expert weights (attention, shared expert, output
     head), which every token reads
   - about 3.6 GB of routed experts

   A desktop with 2-channel DDR5 can deliver at most about 5 tok/s even with
   the whole model in RAM.
2. **Capacity is the first wall on a common PC, and the OS makes it much
   worse.** The model (155 GB) does not fit in 32-64 GB of RAM, so weights
   stream from the SSD through the OS page cache.
   - We measured a performance cliff: with free RAM close to the model size
     (3.4-4.6 GB free for a 3.8 GB model), speed fell from 13.9 to
     0.4-1.3 tok/s.
   - Disk reads reached **2x all the weights a token uses** (3.2x its expert
     weights).
   - The page cache's least-recently-used eviction plus read-ahead thrashes
     under the model's fixed layer order.
   - **This is the largest software-fixable loss**, and nobody has fixed it
     yet in llama.cpp-style engines.
3. **Engines use only half of the memory bandwidth.** Measured decode
   efficiency: 49-54% of DRAM bandwidth at 4-bit, 59-68% at 8-bit. The
   causes:
   - Unpacking 4-bit weights makes each core compute-bound.
   - About 45 µs of fixed overhead per layer (dispatch and thread barriers).
   - A single core cannot saturate DRAM.
4. **Prompt processing is compute-bound and slow:** about 470-520 GFLOP/s on
   4 cores, roughly 15% of the CPU's int8 peak. DeepSeek-V4-Flash needs about
   26 GFLOP per prompt token, so a 2,000-token prompt takes over a minute on a
   desktop even with every weight in RAM.
5. **Long context adds its own bandwidth.** With standard attention, decode
   fell from 21 to 7 tok/s as context grew to 7K tokens. DeepSeek-V4's
   compressed and sparse attention largely removes this; many other models
   do not have it.

## 2. Measurements

### 2.1 Memory bandwidth per core (`tools/membw.c`)

| Threads | Read bandwidth |
|---|---|
| 1 | 9.3-9.5 GB/s |
| 2 | 17.8-18.5 GB/s |
| 3 | 24.4-26.4 GB/s |
| 4 | 32.0-34.6 GB/s |

One core pulls only about 9 GB/s. A core can keep only so many memory
requests in flight, so bandwidth scales with core count until DRAM
saturates.
- On a desktop with 2-channel DDR5 (about 90 GB/s), roughly 4-6 fast cores
  saturate memory.
- On a server with about 460 GB/s, it takes about 40 or more cores.
- On hybrid Intel CPUs, the slower E-cores pull less bandwidth each, and they
  make the fast cores wait at every barrier.

### 2.2 Decode efficiency

Synthetic MoE, 8 layers × 16 experts, 4 active. Each token reads exactly
755 MB at Q4_K or 1,427 MB at Q8_0.

| Quant | Threads | tok/s | Effective GB/s | Share of DRAM bandwidth |
|---|---|---|---|---|
| Q4_K | 1 | 6.97 | 5.3 | 57% |
| Q4_K | 4 | 22.48 | 17.0 | **49%** |
| Q8_0 | 1 | 4.40 | 6.3 | 68% |
| Q8_0 | 4 | 14.31 | 20.4 | **59%** |

The profiler run (4 layers × 8 experts) agrees: Q4_K 51-54%, Q8_0 62-68%.

- **4-bit uses bandwidth less efficiently than 8-bit.** Each core has to
  unpack and scale 4-bit blocks, so per-core throughput is compute-bound
  (about 9.4 G weights/s per core here). Lower-bit formats only pay off if
  enough cores are available to unpack them.
- `-rtr` (repack weights into row-interleaved layouts at load time) gave
  **+11%** decode and +12% prompt speed.

### 2.3 Fixed per-layer overhead

Tiny layers whose weights fit in cache, 8 vs 64 layers, so only dispatch
and synchronization remain:

| Threads | Overhead per layer |
|---|---|
| 1 | 28-36 µs |
| 2-4 | 43-45 µs |

- With tiny layers, 2-4 threads were **slower** than 1 thread: barrier cost
  exceeds the parallel speed-up.
- These layers are simple (llama MoE layer, about 30 graph operations).
  DeepSeek-V4-Flash layers carry compressors, an indexer, sliding-window
  attention and hyper-connections, several times more operations.
- Estimate for V4-Flash: 43 layers × 45-135 µs is about **2-6 ms per token**.
  That is 4-12% of the time budget at 20 tok/s, and it grows with thread
  count.

### 2.4 Prefill compute

| Model | Prompt tok/s (512 tokens, 4 threads) | Compute |
|---|---|---|
| 8-layer MoE, Q4_K | 161.6 (180.5 with `-rtr`) | 434-485 GFLOP/s |
| 8-layer MoE, Q8_0 | 173.1 | 465 GFLOP/s |
| profiler model, Q4_K / Q8_0 | 353 / 390 | 474 / 523 GFLOP/s |

That is about 15-18% of the theoretical int8 peak (about 2.9 TOPS, assuming
two 512-bit VNNI units per core). At about 26 GFLOP per token,
DeepSeek-V4-Flash would prefill at about **18 tok/s on this VM** even with
every weight in RAM.

### 2.5 Context length (standard attention)

Q4_K, 4 threads, KV cache f16 (128 KB per token of context across 8 layers):

| Context already in KV cache | Decode tok/s | Prompt tok/s |
|---|---|---|
| 0 | 21.2 | 158.5 |
| 2,048 | 14.0 | 123.4 |
| 4,096 | 10.9 | 100.1 |
| 7,168 | 7.3 | 76.6 |

At 7K context, reading the KV cache (about 0.9 GB per token) outweighs
reading the weights (0.75 GB). DeepSeek-V4 compresses KV 4:1 (CSA) and 128:1
(HCA) and attends sparsely, which keeps decode cost almost flat with context.
Its indexer still scans all compressed keys, but those are small. Models
without such attention hit this wall on CPUs quickly.

### 2.6 SSD tier: the page-cache cliff

Setup:
- Q8_0 model of 3.8 GB. Each token uses 0.53 GB of non-expert weights plus
  0.86 GB of routed experts.
- Free RAM squeezed with `tools/ram_limit.py`.
- Server run through `deploy/serve.sh` with `EXPERT_STREAMING=on`, i.e.
  `--defer-experts --prefetch-experts`.
- 48-128 generated tokens; disk reads counted from `/proc/vmstat`.
- The disk reads at 1.0 GB/s with 4 MB requests and 0.55 GB/s with 64 KB
  requests.

| Free RAM | tok/s | Disk read per token | Multiple of the ideal miss traffic |
|---|---|---|---|
| plenty | 13.9 | 0 | - |
| 4.6 GB | 1.3 | 704 MB | ~3x (a good cache would miss about 230 MB) |
| 3.4-3.6 GB | **0.4** | **2,750 MB** | about 6x; more than all weights one token needs (1.39 GB) |

Isolating the cause at 3.1 GB free: plain `mmap`, `--defer-experts` alone,
and `--defer-experts --prefetch-experts` all read 2.7-2.9 GB per token at
0.4 tok/s. **It is the page-cache policy, not the engine's prefetcher.**

The mechanism:
- Every token walks layers 0..N in the same order.
- When this loop is larger than the cache, least-recently-used eviction
  removes exactly the pages needed next, so the hit rate falls toward zero.
- Read-ahead and eviction of prefetched-but-not-yet-used pages then multiply
  the disk traffic.
- The non-expert weights, needed by every token, are evicted too.

**Pinning experiment** (`tools/expert_pin.py`): `mlock` the non-expert
weights plus a budget of whole experts, without changing the engine.

| Free RAM | Pinned | tok/s | Disk read per token |
|---|---|---|---|
| 3.6 GB | nothing | 0.4 | 2,746 MB |
| 3.6 GB | non-expert (0.53 GB) | 0.5 | 2,290 MB |
| 3.6 GB | non-expert + 30% of experts (1.5 GB) | **0.8** | **1,524 MB** |
| 4.6 GB | nothing | 1.3 | 704 MB |
| 4.6 GB | 1.5 GB | 1.5 | 626 MB |
| 4.6 GB | 2.0 GB | 1.1 | 895 MB |

- Pinning **doubles** speed when memory is tight.
- Pinning too much **hurts** once the dynamic cache has room, because a
  static pin list cannot follow what is actually hot.
- An explicit, adaptive expert cache should beat both:
  - non-expert weights always resident
  - frequency-aware and resistant to scans
  - exact-slice reads with no read-ahead waste

  For the 3.6 GB case, that projects to about 2 tok/s (0.43 GB of misses per
  token), against 0.4 today.

## 3. Where the time goes: DeepSeek-V4-Flash on the reference PC

Reference PC:
- Core i5 13th gen (6 P-cores)
- 32 GB RAM, about 26 GB inside WSL2
- 2-channel DDR5, about 90 GB/s peak, about 35 GB/s effective at 50% engine
  efficiency
- NVMe at about 5 GB/s

Model: UD-Q4_K_XL, about 155 GB. Each token uses about 3.4 GB of non-expert
weights and 3.6 GB of routed experts.

| Scenario | SSD bytes per token | Time per token | tok/s |
|---|---|---|---|
| Today: page cache, near-zero hits, ~2x amplification (as in 2.6) | ~9-14 GB | ~2-3 s | **~0.3-0.5** |
| Explicit cache: non-expert pinned, no amplification, 12-35% hit rate | 2.4-3.2 GB | 0.6-0.75 s | ~1.3-1.7 |
| + predictive prefetch (SSD reads overlap compute) | same | ~0.5-0.65 s | ~1.5-2 |
| + cache-aware routing lifting hit rate to ~60% | 1.5 GB | ~0.3 s | ~3 |
| + 2-bit experts from SSD (1.8 GB routed per token) at 60% hit rate | 0.7 GB | ~0.2 s | ~5 |
| Reference: whole model in RAM (would need 192 GB) | 0 | ~0.2 s | ~5 |
| Reference: 12-channel server socket | 0 | 26-43 ms | 23-38 |

Prompt processing on the same PC: about 25 tok/s at best, as a compute
limit. So a 2,000-token prompt takes about 80 s, and a 10K-token agent prompt
takes minutes. Streaming adds SSD time, because each 2,048-token batch touches
most experts: about 156 GB per full pass.

## 4. The blockers, ranked

| # | Blocker | Cost on a common PC | Kind | What fixes it | Status |
|---|---|---|---|---|---|
| 1 | Model ≫ RAM, streamed through the OS page cache | 5-10x (0.3-0.5 tok/s instead of ~2-3) | Software | Explicit expert cache (non-expert pinned, frequency-aware, scan-resistant, exact reads), predictive prefetch | **Done: explicit expert cache, 4.4-6x (docs/EXPERT_CACHE.md).** Predictive prefetch open |
| 2 | Low expert-cache hit rate (routing spread over 256 experts) | ~2x at 32 GB | Software (routing) | Cache-aware routing (training-free); ReMoE-style router tuning | Open (docs/HANDOFF.md section 7) |
| 3 | Bytes per token (~7 GB even at 4-bit) | Sets the ~5 tok/s ceiling on 2-channel RAM | Model + format | 2-bit and tiered-precision experts; lower-bit non-expert weights; fewer experts per token (`-ser`) | Partly available; needs quality gates |
| 4 | Engine uses ~50-60% of bandwidth | ~1.7x | Software | Kernels, repacking (+11% measured), fusion, fewer barriers | Partly available |
| 5 | Prompt compute (~15% of int8 peak) | Minutes for long prompts | Software + hardware | Better int8 GEMM (AMX/VNNI), prompt caching, prefill-time expert streaming | Partly available |
| 6 | Per-layer overhead (~45 µs, more on complex layers) | 4-12% at 20 tok/s; worse on many-core | Software | Operator fusion, persistent threads, fewer threads for small ops | Open |
| 7 | Long-context attention traffic | Up to 3x at 7K (standard attention) | Model architecture | Compressed/sparse attention (V4 already has it), KV quantization | Model-dependent |
| 8 | MoE irregularity: speculation and batching touch many more experts | Limits speculation to ~1.1-1.4x | Physics of MoE | Expert-aware drafting and batching | Open |
| 9 | Platform: WSL memory cap and cache dropping, P/E cores, thermals | Varies | Configuration | `.wslconfig`, P-core pinning (handled in `deploy/`) | Done |

## 5. Research agenda

In order of expected gain per effort on common PCs:

1. **Explicit expert cache (R1). Done:** see `docs/EXPERT_CACHE.md`.
   Measured 4.4-6x faster decoding and 5-8x less disk traffic than the page
   cache, with bit-identical outputs. Original design goals:
   - non-expert weights always resident
   - a scan-resistant, frequency-aware pool of whole experts, with a budget
     set from available RAM
   - exact-slice `pread`/`O_DIRECT` reads with no read-ahead
   - misses counted per layer

   Evidence: section 2.6 (2x from static pinning; about 5x projected).
   Target: at least 3x tok/s at 60-90% model-in-RAM on the test VM, with
   disk reads per token under 1.5x the ideal miss traffic.
2. **Cache-aware routing (R2)** on top of R1: a selection-only bonus for
   resident experts, plus optional miss-skip (docs/HANDOFF.md section 7).
   Needs real weights to measure quality (KL divergence).
3. **Predictive prefetch (R3):** use the next layers' routers on the current
   hidden state so SSD reads overlap compute.
4. **Fewer bytes (R4):** tiered precision (2-bit cold experts), lower-bit
   non-expert weights, and measured `-ser`, each behind KL-divergence gates.

## 6. Caveats

- **Routing on random weights** is close to uniform, so the hit rates in 2.6
  are pessimistic. Real models have skewed, temporally correlated routing:
  better for caching, and the reason R2 can work. Re-measure with
  DeepSeek-V2-Lite, OLMoE or V4-Flash itself once weights are reachable.
- **The test VM is not a desktop.** Absolute numbers differ, but the ratios
  (efficiency, overhead, amplification) should carry over. Run
  `tools/engine_profile.py` on the target PC to confirm.
- **Two memory levels.** At the tightest RAM level, the full-pinning case
  could not be measured: the container's out-of-memory killer ended the
  memory holder. Only runs where the holder stayed alive are reported.

## 7. Reproduce

```bash
python3 tools/engine_profile.py --bin <ik_llama.cpp>/build/bin      # sections 2.1-2.5, ~3 min, ~5 GB scratch

# 2.6: SSD tier. Model: python3 deploy/test/make_tiny_gguf.py --embd 4096 --ff 2048 --experts 16 \
#      --used 4 --layers 8 --heads 32 f32.gguf, then llama-quantize --pure f32.gguf q8.gguf Q8_0
python3 tools/ram_limit.py --leave 3.4 &                            # emulate a small-RAM PC
sudo python3 tools/expert_pin.py q8.gguf --budget-gib 1.5 &         # optional: pin
MODEL_FILE=q8.gguf EXPERT_STREAMING=on EXTRA_ARGS="-b 512 -ub 512" deploy/serve.sh &
python3 deploy/smoke_test.py --max-tokens 48 --fixed-length         # read pgpgin in /proc/vmstat before and after
```
