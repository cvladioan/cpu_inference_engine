#!/usr/bin/env python3
"""Back-of-envelope roofline for CPU-only decoding of large MoE models.

Decode on CPU is memory-bandwidth bound: every generated token has to stream
all *active* weights from DRAM once. This script turns model shape, weight
precision and memory bandwidth into rough tokens/s numbers, so hardware and
quantization choices can be compared before any engine code exists.

All model shapes are estimates derived from public specs (total/active params,
experts, top-k). Verify them against each model's config.json before relying
on the output; see docs/PLAN.md for the assumptions.

Usage:
    python3 tools/roofline.py                 # print every table
    python3 tools/roofline.py --eff 0.6       # change engine efficiency
"""

import argparse
from dataclasses import dataclass

GB = 1e9


@dataclass(frozen=True)
class Model:
    name: str
    total_b: float        # total params, billions
    expert_total_b: float  # params in routed experts, billions
    dense_stream_b: float  # non-expert params streamed per token (excl. embedding lookup)
    layers: int            # MoE layers
    experts: int           # routed experts per layer
    top_k: int             # routed experts per token
    active_b: float        # published active params, billions (for FLOPs)

    @property
    def per_expert_b(self) -> float:
        return self.expert_total_b / (self.layers * self.experts) if self.experts else 0.0

    @property
    def routed_active_b(self) -> float:
        return self.per_expert_b * self.layers * self.top_k


# Shapes are estimates; see docs/PLAN.md "Model shapes" for the derivation.
MODELS = {
    # 43 MoE layers, hidden 4096, expert inter 2048, 256 experts, top-6.
    "dsv4-flash": Model("DeepSeek-V4-Flash 284B-A13B", 284, 277.0, 5.98, 43, 256, 6, 13),
    # 58 MoE layers, hidden 7168, expert inter 2048, 256 experts, top-8.
    "dsv3": Model("DeepSeek-V3.x/R1 671B-A37B", 671, 653.9, 15.7, 58, 256, 8, 37),
    # Assumed ~115B in 256 experts x 48 layers, top-8; verify against config.json.
    "qwen35-122b": Model("Qwen3.5-122B-A10B", 122, 115.0, 5.6, 48, 256, 8, 10),
    # Dense 123B for contrast: every weight is read for every token.
    "dense-123b": Model("Dense 123B (contrast)", 123, 0.0, 123.0, 1, 0, 0, 123),
}

# Bits per weight including block scales: (routed experts, everything else).
PRECISIONS = {
    "native": None,  # resolved per model below
    "int8": (8.5, 8.5),
    "mixed": (4.5, 8.5),  # 4-bit experts, 8-bit attention/shared/head
    "q4": (4.5, 4.5),
}
NATIVE = {
    "dsv4-flash": (4.25, 8.25),  # MXFP4 experts, FP8 block-scaled rest
    "dsv3": (8.0, 8.0),          # FP8 block-scaled
    "qwen35-122b": (16.0, 16.0),  # BF16
    "dense-123b": (16.0, 16.0),
}


@dataclass(frozen=True)
class Hardware:
    name: str
    channels: int
    mts: int  # MT/s per channel

    @property
    def peak_gbs(self) -> float:
        return self.channels * self.mts * 8 / 1000  # 64-bit channels

    def sustained_gbs(self, stream_frac: float) -> float:
        return self.peak_gbs * stream_frac


HARDWARE = {
    "tr3995wx": Hardware("TR Pro 3995WX, 8ch DDR4-3200 (calibration)", 8, 3200),
    "genoa": Hardware("EPYC 9004 Genoa, 12ch DDR5-4800", 12, 4800),
    "turin": Hardware("EPYC 9005 Turin, 12ch DDR5-6000", 12, 6000),
    "xeon6": Hardware("Xeon 6 6900P, 12ch DDR5-6400", 12, 6400),
    "xeon6-mr": Hardware("Xeon 6 6900P, 12ch MRDIMM-8800", 12, 8800),
    "venice": Hardware("EPYC 9006 Venice, 16ch DDR5-8000", 16, 8000),
    "venice-mr": Hardware("EPYC 9006 Venice, 16ch MRDIMM-12800", 16, 12800),
}


def bits_for(model_key: str, prec: str) -> tuple:
    return NATIVE[model_key] if prec == "native" else PRECISIONS[prec]


def footprint_gb(m: Model, bits: tuple) -> float:
    be, bd = bits
    return (m.expert_total_b * be + (m.total_b - m.expert_total_b) * bd) / 8


def distinct_experts(m: Model, batch: int) -> float:
    """Expected distinct experts touched per layer by `batch` tokens (uniform routing).

    Real routing is skewed, which makes this pessimistic for batching.
    """
    if not m.experts:
        return 0.0
    return m.experts * (1 - (1 - m.top_k / m.experts) ** batch)


def step_gb(m: Model, bits: tuple, batch: int) -> float:
    """Bytes streamed for one forward step over `batch` tokens (decode or verify)."""
    be, bd = bits
    dense = m.dense_stream_b * bd / 8
    experts = m.layers * distinct_experts(m, batch) * m.per_expert_b * be / 8
    return dense + experts


def decode_tps(m: Model, bits: tuple, bw_gbs: float, eff: float, batch: int = 1) -> float:
    """Aggregate decode tokens/s for `batch` concurrent streams (bandwidth bound)."""
    return batch * bw_gbs * eff / step_gb(m, bits, batch)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream-frac", type=float, default=0.78,
                    help="sustained/peak DRAM bandwidth (STREAM triad), default 0.78")
    ap.add_argument("--eff", type=float, default=0.75,
                    help="target engine efficiency: share of sustained bandwidth spent streaming weights")
    ap.add_argument("--eff-today", type=float, default=0.45,
                    help="efficiency of today's engines (calibrated on ik_llama.cpp, 3995WX)")
    ap.add_argument("--target-tps", type=float, default=0,
                    help="only print the memory bandwidth a single stream needs for this tok/s")
    args = ap.parse_args()

    if args.target_tps:
        print_target(args)
        return

    print("## Weight footprint and bytes streamed per decoded token\n")
    rows = []
    for mk, m in MODELS.items():
        for pk in PRECISIONS:
            if m.experts == 0 and pk == "mixed":
                continue
            b = bits_for(mk, pk)
            rows.append([m.name, pk, f"{b[0]}/{b[1]}", f"{footprint_gb(m, b):.0f}", f"{step_gb(m, b, 1):.1f}"])
    print(md_table(["model", "precision", "bpw experts/rest", "weights GB", "GB per token"], rows))

    print(f"\n## Single-stream decode tok/s per socket (today eff={args.eff_today}, target eff={args.eff})\n")
    cases = [("dsv4-flash", "native"), ("dsv4-flash", "q4"), ("qwen35-122b", "mixed"),
             ("qwen35-122b", "q4"), ("dsv3", "mixed"), ("dense-123b", "q4")]
    headers = ["hardware", "sustained GB/s"] + [f"{MODELS[mk].name.split(' ')[0]} {pk}" for mk, pk in cases]
    rows = []
    for hw in HARDWARE.values():
        bw = hw.sustained_gbs(args.stream_frac)
        cells = []
        for mk, pk in cases:
            m, b = MODELS[mk], bits_for(mk, pk)
            cells.append(f"{decode_tps(m, b, bw, args.eff_today):.0f}-{decode_tps(m, b, bw, args.eff):.0f}")
        rows.append([hw.name, f"{bw:.0f}"] + cells)
    print(md_table(headers, rows))

    print("\n## Batched decode, DeepSeek-V4-Flash native, one Xeon 6 MRDIMM socket, target eff\n")
    m, b = MODELS["dsv4-flash"], NATIVE["dsv4-flash"]
    bw = HARDWARE["xeon6-mr"].sustained_gbs(args.stream_frac)
    rows = []
    for batch in (1, 2, 4, 8, 16, 32, 64, 128):
        agg = decode_tps(m, b, bw, args.eff, batch)
        rows.append([batch, f"{distinct_experts(m, batch):.0f}", f"{step_gb(m, b, batch):.0f}",
                     f"{agg:.0f}", f"{agg / batch:.1f}"])
    print(md_table(["concurrent streams", "distinct experts/layer", "GB per step",
                    "aggregate tok/s", "tok/s per stream"], rows))

    print("\n## MTP speculative decoding speedup (bandwidth model), DeepSeek-V4-Flash native\n")
    rows = []
    base = step_gb(m, b, 1)
    for draft in (1, 2, 3):
        cost = step_gb(m, b, draft + 1) / base
        cells = [draft, f"{cost:.2f}x"]
        for alpha in (0.7, 0.8, 0.9):
            expected = (1 - alpha ** (draft + 1)) / (1 - alpha)
            cells.append(f"{expected / cost:.2f}x")
        rows.append(cells)
    print(md_table(["draft tokens", "verify cost vs 1 token", "speedup a=0.7", "a=0.8", "a=0.9"], rows))

    print("\n## Prefill tok/s ceiling from compute (2 x active params FLOPs/token, attention ignored)\n")
    rows = []
    for mk in ("qwen35-122b", "dsv4-flash", "dsv3"):
        mm = MODELS[mk]
        rows.append([mm.name] + [f"{t * 1e12 / (2 * mm.active_b * 1e9):.0f}" for t in (10, 25, 50, 100)])
    print(md_table(["model", "10 TOPS eff.", "25", "50", "100"], rows))

    print_ssd_streaming(args)


def print_target(args):
    """Peak DRAM bandwidth one socket needs for `args.target_tps` single-stream tok/s."""
    t = args.target_tps
    print(f"## Memory bandwidth for {t:g} tok/s single-stream decode (DeepSeek-V4-Flash, all weights in RAM)\n")
    rows = []
    for label, bits in (("UD-Q8_K_XL (native experts, 8-bit rest)", NATIVE["dsv4-flash"]),
                        ("UD-Q4_K_XL (4-bit)", PRECISIONS["q4"])):
        gb = step_gb(MODELS["dsv4-flash"], bits, 1)
        cells = [label, f"{gb:.1f}"]
        for eff in (args.eff_today, 0.6, args.eff):
            cells.append(f"{t * gb / (args.stream_frac * eff):.0f}")
        rows.append(cells)
    print(md_table(["quant", "GB per token", f"peak GB/s at eff {args.eff_today}", "at eff 0.6",
                    f"at eff {args.eff}"], rows))
    need = t * step_gb(MODELS["dsv4-flash"], PRECISIONS["q4"], 1) / (args.stream_frac * args.eff_today)
    print(f"\nPlatforms (per socket, all channels populated) vs {need:.0f} GB/s for 4-bit at today's efficiency:\n")
    rows = [[hw.name, f"{hw.peak_gbs:.0f}", "yes" if hw.peak_gbs >= need else "no"] for hw in HARDWARE.values()]
    rows.append(["Desktop, 2ch DDR5-5600", f"{2 * 5600 * 8 / 1000:.0f}", "no"])
    print(md_table(["platform", "peak GB/s", f"reaches {t:g} tok/s"], rows))


def ssd_stream_tps(m: Model, bits: tuple, ram_gbs: float, ssd_gbs: float, hit: float) -> float:
    """Single-stream decode tok/s when routed experts live on SSD with a RAM cache.

    Dense weights are RAM-resident; a fraction `hit` of routed-expert bytes comes
    from the RAM cache, the rest from SSD. Assumes no overlap of SSD reads with
    compute (prefetching can hide part of it), so this is the conservative case.
    """
    be, bd = bits
    dense = m.dense_stream_b * bd / 8
    routed = m.routed_active_b * be / 8
    t = (dense + hit * routed) / ram_gbs + (1 - hit) * routed / ssd_gbs
    return 1 / t


def print_ssd_streaming(args):
    m, bits = MODELS["dsv4-flash"], PRECISIONS["q4"]
    expert_gb = m.expert_total_b * bits[0] / 8
    dense_gb = (m.total_b - m.expert_total_b) * bits[1] / 8
    # Dual-channel DDR5-6000 desktop; `eff` share of sustained bandwidth.
    ram = Hardware("desktop, 2ch DDR5-6000", 2, 6000).sustained_gbs(args.stream_frac) * args.eff_today
    print(f"\n## SSD expert streaming: DeepSeek-V4-Flash 4-bit ({dense_gb + expert_gb:.0f} GB) on a desktop "
          f"(2ch DDR5-6000, {ram:.0f} GB/s effective)\n")
    print("Expert cache = RAM minus ~12 GB (OS, KV cache) minus the dense weights. Uniform routing would give a hit")
    print("rate equal to the cache fraction; real routing is skewed, so measure it.\n")
    rows = []
    for ram_gb in (32, 64, 96, 128):
        cache = max(0.0, min(1.0, (ram_gb - 12 - dense_gb) / expert_gb))
        rows.append([f"{ram_gb} GB", f"{cache:.0%}"])
    print(md_table(["RAM", "share of experts cached"], rows))
    print()
    # NVMe read speed for multi-MB expert chunks, ~80% of sequential.
    ssds = [("PCIe 4.0 x4 (~7 GB/s)", 5.6), ("PCIe 5.0 x4 (~14 GB/s)", 11.2), ("2x PCIe 5.0 RAID0", 22.4)]
    hits = (0.3, 0.5, 0.7, 0.9)
    rows = [[name] + [f"{ssd_stream_tps(m, bits, ram, bw, h):.1f}" for h in hits] for name, bw in ssds]
    rows.append(["all in RAM (reference)"] + [f"{ssd_stream_tps(m, bits, ram, 1e9, 1.0):.1f}"] * len(hits))
    print(md_table(["SSD"] + [f"tok/s at {h:.0%} hit" for h in hits], rows))


if __name__ == "__main__":
    main()
