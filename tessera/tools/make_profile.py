#!/usr/bin/env python3
"""Build a hot-expert profile from routing traces: every (layer, expert) pair, most used first.

A trace is written by the engine when TESSERA_ROUTING_TRACE=<file> is set (scripts/calibrate.sh does it):
records of int32 layer, int32 n_tokens, int32 k, then n_tokens * k int32 expert ids.

Only decode steps (at most --max-tokens tokens per record) are counted by default: the hot experts exist for
decoding, while prompts are processed in large batches that stream every expert anyway. Pairs never seen follow,
interleaved across layers, so any budget can be filled.

    python3 tessera/tools/make_profile.py trace.bin [more.bin ...] --out model.profile [--n-layer 48 --n-expert 512]

Output: text, one "layer expert" pair per line (what the engine's --hot-experts reads). Standard library only.
"""

import argparse
import struct
import sys
from collections import Counter


def read_trace(path, max_tokens):
    """Counter of (layer, expert) over decode records, and the same over all records."""
    decode, every = Counter(), Counter()
    layers, max_e = set(), -1
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    while off + 12 <= len(data):
        layer, n_tok, k = struct.unpack_from("<iii", data, off)
        off += 12
        n = n_tok * k
        if n < 0 or off + 4 * n > len(data):
            print(f"{path}: truncated record at byte {off - 12}, stopping there", file=sys.stderr)
            break
        ids = struct.unpack_from(f"<{n}i", data, off)
        off += 4 * n
        layers.add(layer)
        for e in ids:
            if e < 0:
                continue
            max_e = max(max_e, e)
            every[(layer, e)] += 1
            if n_tok <= max_tokens:
                decode[(layer, e)] += 1
    return decode, every, layers, max_e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8, help="records with at most this many tokens are decode steps")
    ap.add_argument("--n-layer", type=int, help="layers (default: from the traces)")
    ap.add_argument("--n-expert", type=int, help="experts per layer (default: from the traces)")
    args = ap.parse_args()

    decode, every, layers, max_e = Counter(), Counter(), set(), -1
    for t in args.traces:
        d, a, ls, me = read_trace(t, args.max_tokens)
        decode.update(d)
        every.update(a)
        layers |= ls
        max_e = max(max_e, me)
    counts = decode if decode else every
    if not counts:
        sys.exit("no routed experts in the traces")
    n_layer = args.n_layer or (max(layers) + 1)
    n_expert = args.n_expert or (max_e + 1)
    moe_layers = sorted(layers) if not args.n_layer else range(n_layer)

    ranked = [pair for pair, _ in counts.most_common()]
    seen = set(ranked)
    # never routed: round-robin over layers, so a big budget still spreads evenly
    for e in range(n_expert):
        for layer in moe_layers:
            if (layer, e) not in seen:
                ranked.append((layer, e))

    total = sum(counts.values())
    with open(args.out, "w") as f:
        f.write(f"# Tessera hot-expert profile: {len(counts)} routed pairs seen in {total} expert uses "
                f"({'decode' if decode else 'all'} records of {len(args.traces)} trace(s)); most used first\n")
        for layer, e in ranked:
            f.write(f"{layer} {e}\n")

    # how much of the traffic the top experts carry: the curve that decides the VRAM budget
    all_pairs = len(moe_layers) * n_expert
    print(f"{args.out}: {len(ranked)} pairs, {len(counts)} of {all_pairs} used in the traces")
    top = [c for _, c in counts.most_common()]
    for share in (0.05, 0.1, 0.2, 0.3, 0.5):
        n = int(all_pairs * share)
        print(f"  top {int(share * 100):2d}% of experts ({n:6d}) serve {100 * sum(top[:n]) / total:5.1f}% of the uses")


if __name__ == "__main__":
    main()
