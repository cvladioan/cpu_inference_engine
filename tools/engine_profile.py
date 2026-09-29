#!/usr/bin/env python3
"""Measure what limits LLM inference on this CPU (see docs/BOTTLENECKS.md).

Runs, on the local machine:
  1. DRAM read bandwidth per thread count (tools/membw.c)
  2. decode efficiency: tok/s x bytes-per-token / bandwidth, on a synthetic MoE
     with DeepSeek-V4-Flash-sized experts (hidden 4096, expert width 2048),
     quantized to Q4_K and Q8_0
  3. prefill (prompt) compute throughput in GFLOP/s
  4. fixed per-layer overhead (graph dispatch + thread sync) with tiny layers

Needs an ik_llama.cpp build with llama-bench and llama-quantize (deploy/build.sh)
and gcc with OpenMP. Writes ~5 GB of temporary models into --work.

    python3 tools/engine_profile.py
    python3 tools/engine_profile.py --threads "1 2 4 6" --bin ~/deepseek-cpu/ik_llama.cpp/build/bin
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "deploy" / "test" / "make_tiny_gguf.py"
BPW = {"Q4_K": 4.5, "Q8_0": 8.5}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def physical_cores():
    out = run(["lscpu", "-p=CORE,SOCKET"])
    return len({line for line in out.splitlines() if not line.startswith("#")})


def membw(work, threads):
    exe = work / "membw"
    run(["gcc", "-O3", "-march=native", "-fopenmp", str(ROOT / "tools" / "membw.c"), "-o", str(exe)])
    out = run([str(exe), "2", str(max(threads))])
    return {int(t): float(g) for t, g in re.findall(r"^\s*(\d+)\s+([\d.]+)\s*$", out, re.M)}


def bench(bin_dir, model, threads, n_prompt, n_gen, reps=3, extra=()):
    out = run([str(bin_dir / "llama-bench"), "-m", str(model), "-t", ",".join(map(str, threads)),
               "-p", str(n_prompt), "-n", str(n_gen), "-r", str(reps), "-o", "json", *extra], cwd=model.parent)
    return {r["n_threads"]: r["avg_ts"] for r in json.loads(out)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.expanduser("~/deepseek-cpu/ik_llama.cpp/build/bin"),
                    help="directory with llama-bench and llama-quantize")
    ap.add_argument("--threads", default="", help='thread counts, e.g. "1 2 4 6" (default: powers of 2 + all cores)')
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--used", type=int, default=4)
    ap.add_argument("--work", default="", help="scratch directory (default: a temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the generated models")
    args = ap.parse_args()

    bin_dir = Path(args.bin)
    for tool in ("llama-bench", "llama-quantize"):
        if not (bin_dir / tool).exists():
            sys.exit(f"{bin_dir / tool} not found; build it (deploy/build.sh) or pass --bin")
    cores = physical_cores()
    threads = [int(t) for t in args.threads.split()] if args.threads else \
        sorted({t for t in (1, 2, 4, 8, 16, 32, 64) if t < cores} | {cores})
    work = Path(args.work or tempfile.mkdtemp(prefix="engine_profile-"))
    work.mkdir(parents=True, exist_ok=True)
    results = {"cpu": run(["lscpu"]).split("Model name:")[1].splitlines()[0].strip(), "threads": threads}
    print(f"CPU: {results['cpu']} ({cores} physical cores); scratch: {work}", flush=True)

    print("\n[1/4] DRAM read bandwidth", flush=True)
    bw = membw(work, threads)
    results["bandwidth_gbs"] = bw
    print("  " + ", ".join(f"{t} thr: {g:.1f} GB/s" for t, g in bw.items()))

    print("\n[2/4] decode efficiency on a V4-Flash-shaped MoE", flush=True)
    e, ff, L, E, U = 4096, 2048, args.layers, args.experts, args.used
    active = L * (4 * e * e + e * E + U * 3 * e * ff)  # attention + router + active experts
    f32 = work / "moe-f32.gguf"
    run([sys.executable, str(GEN), "--embd", str(e), "--ff", str(ff), "--experts", str(E), "--used", str(U),
         "--layers", str(L), "--heads", "32", str(f32)])
    models = {}
    for q in BPW:
        models[q] = work / f"moe-{q}.gguf"
        run([str(bin_dir / "llama-quantize"), "--pure", str(f32), str(models[q]), q, str(max(threads))])
    f32.unlink()
    results["decode"] = {}
    for q, path in models.items():
        gb_tok = active * BPW[q] / 8 / 1e9
        tps = bench(bin_dir, path, threads, 0, 64)
        rows = {t: {"tok_s": v, "gbs": v * gb_tok, "eff": v * gb_tok / bw[t] if t in bw else None}
                for t, v in tps.items()}
        results["decode"][q] = {"gb_per_token": gb_tok, "by_threads": rows}
        for t, r in rows.items():
            eff = f"{r['eff']:.0%}" if r["eff"] else "n/a"
            print(f"  {q} {t:>3} thr: {r['tok_s']:6.1f} tok/s = {r['gbs']:5.1f} GB/s effective ({eff} of DRAM)")

    print("\n[3/4] prefill compute", flush=True)
    results["prefill"] = {}
    for q, path in models.items():
        pp = bench(bin_dir, path, [max(threads)], 512, 0, reps=2)[max(threads)]
        gflops = pp * 2 * active / 1e9
        results["prefill"][q] = {"tok_s": pp, "gflops": gflops}
        print(f"  {q}: {pp:.0f} tok/s = {gflops:.0f} GFLOP/s")

    print("\n[4/4] fixed per-layer overhead (tiny cache-resident layers)", flush=True)
    per = {}
    for layers in (8, 64):
        path = work / f"tiny-{layers}.gguf"
        run([sys.executable, str(GEN), "--embd", "64", "--ff", "64", "--experts", "8", "--used", "2",
             "--layers", str(layers), "--heads", "1", str(path)])
        per[layers] = bench(bin_dir, path, [1, max(threads)], 0, 256)
    results["overhead_us_per_layer"] = {
        t: (1e6 / per[64][t] - 1e6 / per[8][t]) / 56 for t in per[8]}
    for t, us in results["overhead_us_per_layer"].items():
        print(f"  {t:>3} thr: {us:.1f} us per layer")

    out = work / "engine_profile.json"
    out.write_text(json.dumps(results, indent=2))
    if not args.keep:
        for p in work.glob("*.gguf"):
            p.unlink()
    print(f"\nsaved {out}")
    best = max(results["decode"]["Q4_K"]["by_threads"].values(), key=lambda r: r["tok_s"])
    print("\nsummary:")
    print(f"  DRAM bandwidth: {max(bw.values()):.1f} GB/s")
    print(f"  best decode efficiency: Q4_K {best['eff']:.0%}, "
          f"Q8_0 {max(r['eff'] or 0 for r in results['decode']['Q8_0']['by_threads'].values()):.0%}")
    print(f"  prefill: {results['prefill']['Q4_K']['gflops']:.0f} GFLOP/s -> DeepSeek-V4-Flash (26 GFLOP/token) "
          f"prompt speed ~{results['prefill']['Q4_K']['gflops'] / 26:.0f} tok/s if all weights were in RAM")
    print(f"  per-layer overhead: {results['overhead_us_per_layer'][max(threads)]:.0f} us at {max(threads)} threads")


if __name__ == "__main__":
    main()
