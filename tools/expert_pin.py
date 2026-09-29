#!/usr/bin/env python3
"""Pin the always-used weights and a budget of experts in RAM for SSD-streamed MoE models.

Why: when a MoE model is larger than RAM, llama.cpp-style engines leave the
weights to the OS page cache. Every token walks the layers in the same order,
so once the working set is a bit larger than the cache, least-recently-used
eviction throws out exactly the pages needed next: the hit rate collapses and
read-ahead multiplies disk traffic (docs/BOTTLENECKS.md). Pinning part of the
file turns that cliff into a proportional hit rate.

How: maps the GGUF file read-only and mlock()s
  * every non-expert tensor except the token embedding (read one row at a time),
  * then whole experts (gate/up/down slices of the same expert together),
    spread round-robin over layers, until --budget-gib is used.
Page-cache pages are shared, so the inference server reading the same file
gets these from RAM. Runs until killed; needs root or a raised memlock limit.

    sudo python3 tools/expert_pin.py model.gguf --budget-gib 2 &
    deploy/serve.sh                 # EXPERT_STREAMING=on

--hot FILE takes a JSON {"layer": [expert ids by descending use]} to pin the
most used experts first instead of round-robin ids. Standard library only.
"""

import argparse
import ctypes
import json
import mmap
import os
import re
import signal
import struct
import sys
import time

PAGE = mmap.PAGESIZE
# GGUF metadata value types -> struct format (fixed-size ones)
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.f.read(size))

    def string(self):
        (n,) = self.unpack("Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in SCALAR:
            return self.unpack(SCALAR[vtype])[0]
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, count = self.unpack("IQ")
            if etype in SCALAR:  # skip fixed-size arrays without decoding them
                self.f.seek(struct.calcsize("<" + SCALAR[etype]) * count, os.SEEK_CUR)
                return None
            return [self.value(etype) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf(path):
    """Tensor table: name -> (dims, absolute byte offset, byte size)."""
    with open(path, "rb") as f:
        r = Reader(f)
        magic, version, n_tensors, n_kv = r.unpack("4sIQQ")
        if magic != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            (vtype,) = r.unpack("I")
            val = r.value(vtype)
            if key == "general.alignment":
                alignment = val
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            (n_dims,) = r.unpack("I")
            dims = r.unpack("Q" * n_dims)
            _, offset = r.unpack("IQ")
            infos.append((name, dims, offset))
        data_start = -(-f.tell() // alignment) * alignment
    file_size = os.path.getsize(path)
    # Sizes from consecutive offsets: no need to know every quantization format.
    order = sorted(infos, key=lambda t: t[2])
    tensors = {}
    for i, (name, dims, off) in enumerate(order):
        end = order[i + 1][2] if i + 1 < len(order) else file_size - data_start
        tensors[name] = (dims, data_start + off, end - off)
    return tensors


def plan(tensors, budget, hot=None):
    """Byte ranges to pin: dense tensors first, then experts round-robin (or hottest first)."""
    ranges, used = [], 0
    experts = {}  # layer -> [(tensor name, n_expert)]
    for name, (dims, off, size) in sorted(tensors.items(), key=lambda t: t[1][1]):
        if "_exps" in name and len(dims) == 3:
            layer = int(re.match(r"blk\.(\d+)\.", name).group(1))
            experts.setdefault(layer, []).append(name)
        elif not name.startswith("token_embd") and used + size <= budget:
            ranges.append((off, size))
            used += size
    dense = used

    # Candidate (layer, expert id) order: hottest first per layer, interleaved across layers.
    per_layer = {}
    for layer, names in experts.items():
        n_expert = tensors[names[0]][0][2]
        ids = [int(e) for e in hot.get(str(layer), [])] if hot else []
        ids += [e for e in range(n_expert) if e not in ids]
        per_layer[layer] = ids
    rank = 0
    while True:
        progressed = False
        for layer in sorted(per_layer):
            ids = per_layer[layer]
            if rank >= len(ids):
                continue
            progressed = True
            e = ids[rank]
            slices = []
            for name in experts[layer]:
                dims, off, size = tensors[name]
                per = size // dims[2]
                slices.append((off + e * per, per))
            cost = sum(s for _, s in slices)
            if used + cost > budget:
                return ranges, dense, used
            ranges += slices
            used += cost
        if not progressed:
            return ranges, dense, used
        rank += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="GGUF file (first shard only is supported)")
    ap.add_argument("--budget-gib", type=float, required=True, help="RAM to pin")
    ap.add_argument("--hot", help="JSON file with expert ids per layer, most used first")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without pinning")
    args = ap.parse_args()

    tensors = read_gguf(args.model)
    hot = json.load(open(args.hot)) if args.hot else None
    ranges, dense, used = plan(tensors, int(args.budget_gib * (1 << 30)), hot)
    n_exp_total = sum(size for n, (d, o, size) in tensors.items() if "_exps" in n)
    print(f"plan: {dense / 2**30:.2f} GiB non-expert weights + {(used - dense) / 2**30:.2f} GiB experts "
          f"({(used - dense) / max(1, n_exp_total):.0%} of all expert bytes)", flush=True)
    if args.dry_run:
        return

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    size = os.path.getsize(args.model)
    fd = os.open(args.model, os.O_RDONLY)
    base = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if base in (None, ctypes.c_void_p(-1).value):
        sys.exit(f"mmap failed: {os.strerror(ctypes.get_errno())}")
    t0 = time.time()
    locked = 0
    for off, length in ranges:
        start = off - off % PAGE
        end = off + length
        if libc.mlock(ctypes.c_void_p(base + start), end - start) != 0:
            sys.exit(f"mlock failed after {locked / 2**30:.2f} GiB: {os.strerror(ctypes.get_errno())} "
                     "(run as root or raise the memlock limit)")
        locked += end - start
    print(f"pinned {locked / 2**30:.2f} GiB in {time.time() - t0:.1f} s; holding until killed", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
