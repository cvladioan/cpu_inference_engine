#!/usr/bin/env python3
"""Plan where a MoE model's weights go on this PC, and estimate its speed.

Tessera's tiers (docs/PLAN.md):
  * VRAM: everything every token uses (attention, shared experts, output head, KV cache) plus as many of the
    most-used routed experts as fit ("hot experts", --hot-experts-mib);
  * RAM:  all routed experts, memory-mapped from the model file; the CPU computes the ones that are not hot;
  * SSD:  when the experts do not fit in RAM, an explicit expert cache keeps the used ones (--expert-cache).

    python3 tessera/tools/plan.py model.gguf                      # detects VRAM (nvidia-smi) and free RAM
    python3 tessera/tools/plan.py model.gguf --vram-gib 12 --ram-gib 26 --ctx 32768 --estimate
    python3 tessera/tools/plan.py model.gguf --env                # KEY=VALUE lines for scripts/serve.sh

The estimate is a bandwidth model, not a measurement: use it to compare options, then measure (scripts/bench.sh).
Standard library only.
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_info import kv_bytes, model_info  # noqa: E402

GIB = 1 << 30
MIB = 1 << 20

# Memory bandwidth (GB/s) of common cards, for the estimate; --gpu-bw overrides.
GPU_BW = {
    "3060": 360, "3060 ti": 448, "3070": 448, "3070 ti": 608, "3080": 760, "3090": 936,
    "4060": 272, "4060 ti": 288, "4070": 504, "4070 super": 504, "4070 ti": 504, "4070 ti super": 672,
    "4080": 717, "4090": 1008, "5060 ti": 448, "5070": 672, "5070 ti": 896, "5080": 960, "5090": 1792,
}


def detect_gpu():
    """(name, total VRAM in GiB) of GPU 0, or (None, 0)."""
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    if not smi:
        return None, 0.0
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        name, mib = out[0].rsplit(",", 1)
        return name.strip(), float(mib) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None, 0.0


def gpu_bandwidth(name):
    if not name:
        return 450
    low = name.lower()
    best = None
    for key, bw in GPU_BW.items():
        if key in low and (best is None or len(key) > len(best[0])):
            best = (key, bw)
    return best[1] if best else 450


def mem_available_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return 0.0


def plan(info, vram_gib, ram_gib, ctx, ubatch, vram_reserve_gib, ram_reserve_gib):
    """Byte budgets for each tier."""
    kv = kv_bytes(info, ctx)
    largest_expert_tensor = max(info["expert_bytes_per_layer_expert"].values() or [0]) * info["n_expert"] // 3
    # activations for a ubatch, plus one layer's expert tensor that prompt processing copies to the GPU
    compute = int(0.3 * GIB + ubatch * 0.6 * MIB + largest_expert_tensor)
    vram = int(vram_gib * GIB)
    hot = vram - int(vram_reserve_gib * GIB) - info["dense_bytes"] - kv - compute - int(0.3 * GIB)
    hot = max(0, min(hot, info["expert_bytes"]))
    ram = int(ram_gib * GIB) - int(ram_reserve_gib * GIB)
    ram_experts = ram - info["embedding_bytes"]
    # the CPU reads only the experts that are not hot, but prompt processing touches all of them
    cache = 0 if info["expert_bytes"] <= ram_experts else max(0, ram_experts)
    return {"kv": kv, "compute": compute, "hot": hot, "ram": ram, "cache": cache}


def estimate(info, p, gpu_bw, ram_bw, ssd_bw, hit=None):
    """Decode tok/s without speculation: max(GPU, CPU) per token plus fixed per-layer costs."""
    if not info["n_expert_used"] or not info["expert_bytes"]:
        return None
    per_tok = info["n_expert_used"] * info["moe_layers"] * info["expert_bytes_per_expert"]
    coverage = p["hot"] / info["expert_bytes"]
    # A static profile on real routing (Strata's measurements: 13% of the experts serve ~50% of the reads).
    h = hit if hit is not None else min(0.97, coverage ** 0.35) if coverage > 0 else 0.0
    gpu_ms = (info["dense_bytes"] + h * per_tok) / (gpu_bw * 0.5e9) * 1e3
    cpu_ms = (1 - h) * per_tok / (min(ram_bw * 0.6, 22.0) * 1e9) * 1e3
    ssd_ms = 0.0
    if p["cache"]:
        cold = info["expert_bytes"] - p["hot"]
        in_ram = min(1.0, p["cache"] / cold) if cold else 1.0
        ssd_ms = (1 - h) * per_tok * (1 - in_ram ** 0.35) / (ssd_bw * 1e9) * 1e3
    overhead_ms = info["moe_layers"] * 0.12 + 1.5
    ms = max(gpu_ms, cpu_ms + ssd_ms) + overhead_ms
    return {"hit": h, "gpu_ms": gpu_ms, "cpu_ms": cpu_ms, "ssd_ms": ssd_ms, "overhead_ms": overhead_ms,
            "tok_s": 1000 / ms, "per_token_mib": per_tok / MIB}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--vram-gib", type=float, help="GPU memory (default: nvidia-smi)")
    ap.add_argument("--ram-gib", type=float, help="RAM available to the engine (default: MemAvailable now)")
    ap.add_argument("--ctx", type=int, default=32768, help="context tokens (default 32768)")
    ap.add_argument("--ubatch", type=int, default=1024, help="prompt micro-batch (default 1024)")
    ap.add_argument("--vram-reserve-gib", type=float, default=1.2,
                    help="VRAM for the CUDA context and, on Windows, the desktop (default 1.2)")
    ap.add_argument("--ram-reserve-gib", type=float, default=2.5, help="RAM left to the OS (default 2.5)")
    ap.add_argument("--estimate", action="store_true", help="print a decode speed estimate")
    ap.add_argument("--gpu-bw", type=float, help="GPU memory bandwidth, GB/s (default: from the card's name)")
    ap.add_argument("--ram-bw", type=float, default=40.0, help="RAM bandwidth, GB/s (default 40: dual-channel DDR4)")
    ap.add_argument("--ssd-bw", type=float, default=1.0, help="SSD random 1 MB read speed, GB/s (default 1.0)")
    ap.add_argument("--hit", type=float, help="measured VRAM hit rate, instead of the model's guess")
    ap.add_argument("--env", action="store_true", help="print HOT_MIB=.. CACHE_MIB=.. for scripts")
    args = ap.parse_args()

    info = model_info(args.model)
    gpu_name, vram = detect_gpu()
    vram = args.vram_gib if args.vram_gib is not None else vram
    ram = args.ram_gib if args.ram_gib is not None else mem_available_gib()
    p = plan(info, vram, ram, args.ctx, args.ubatch, args.vram_reserve_gib, args.ram_reserve_gib)

    if args.env:
        print(f"HOT_MIB={p['hot'] // MIB}")
        print(f"CACHE_MIB={p['cache'] // MIB}")
        print(f"KV_GIB={p['kv'] / GIB:.2f}")
        return
    print(f"model  {os.path.basename(args.model)}: {info['file_bytes'] / GIB:.1f} GiB, {info['arch']}, "
          f"{info['n_expert']} experts x {info['moe_layers']} layers, {info['n_expert_used']} per token")
    print(f"GPU    {gpu_name or 'not detected'}, {vram:.1f} GiB  |  RAM for the engine {ram:.1f} GiB")
    print(f"VRAM   dense {info['dense_bytes'] / GIB:.2f} + KV {p['kv'] / GIB:.2f} (ctx {args.ctx}) + compute "
          f"{p['compute'] / GIB:.2f} + reserve {args.vram_reserve_gib:.1f} GiB  ->  hot experts {p['hot'] / GIB:.2f} GiB "
          f"({100 * p['hot'] / max(1, info['expert_bytes']):.0f}% of {info['expert_bytes'] / GIB:.1f} GiB)")
    if p["hot"] <= 0:
        print("       nothing left for hot experts: lower --ctx, or the dense part alone does not fit this card")
    if p["cache"]:
        print(f"RAM    experts do not fit: expert cache {p['cache'] / GIB:.1f} GiB, the rest read from the SSD "
              f"(--defer-experts --expert-cache {p['cache'] // MIB})")
    else:
        print(f"RAM    all experts fit ({info['expert_bytes'] / GIB:.1f} of {p['ram'] / GIB:.1f} GiB): no SSD tier")
    if args.estimate:
        e = estimate(info, p, args.gpu_bw or gpu_bandwidth(gpu_name), args.ram_bw, args.ssd_bw, args.hit)
        if e:
            print(f"speed  ~{e['tok_s']:.0f} tok/s decode, no speculation (rough): VRAM hit {100 * e['hit']:.0f}%, "
                  f"GPU {e['gpu_ms']:.1f} ms | CPU {e['cpu_ms']:.1f} ms"
                  + (f" + SSD {e['ssd_ms']:.1f} ms" if e["ssd_ms"] else "")
                  + f" | fixed {e['overhead_ms']:.1f} ms per token; {e['per_token_mib']:.0f} MiB of experts per token")


if __name__ == "__main__":
    main()
