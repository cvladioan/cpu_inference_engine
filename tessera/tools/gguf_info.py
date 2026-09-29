#!/usr/bin/env python3
"""What a GGUF model needs from each memory tier: dense weights, experts per layer, KV cache shape.

Reads only the header (metadata and tensor table), so it is instant on a 60 GB model. Split models
(`name-00001-of-00003.gguf`) are read across all their shards.

    python3 tessera/tools/gguf_info.py model.gguf          # human summary
    python3 tessera/tools/gguf_info.py model.gguf --json   # for scripts

Standard library only.
"""

import argparse
import json
import os
import re
import struct
import sys

SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
EXPERT_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down|up_gate)_exps\.weight$")


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
            if etype in SCALAR:
                size = struct.calcsize("<" + SCALAR[etype])
                if count <= 4096:   # small arrays (per-layer settings) are kept
                    return list(struct.unpack("<" + SCALAR[etype] * count, self.f.read(size * count)))
                self.f.seek(size * count, os.SEEK_CUR)
                return None
            return [self.value(etype) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")


def shards(path):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(3))
    return [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, n + 1)]


def read_one(path, meta, tensors):
    with open(path, "rb") as f:
        r = Reader(f)
        magic, _version, n_tensors, n_kv = r.unpack("4sIQQ")
        if magic != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            (vtype,) = r.unpack("I")
            val = r.value(vtype)
            meta.setdefault(key, val)
            if key == "general.alignment":
                alignment = val
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            (n_dims,) = r.unpack("I")
            dims = r.unpack("Q" * n_dims)
            _type, offset = r.unpack("IQ")
            infos.append((name, dims, offset))
        data_start = -(-f.tell() // alignment) * alignment
    size = os.path.getsize(path)
    order = sorted(infos, key=lambda t: t[2])
    for i, (name, dims, off) in enumerate(order):
        end = order[i + 1][2] if i + 1 < len(order) else size - data_start
        tensors[name] = {"dims": list(dims), "bytes": end - off, "file": os.path.basename(path)}


def model_info(path):
    meta, tensors = {}, {}
    for p in shards(path):
        read_one(p, meta, tensors)
    arch = meta.get("general.architecture", "?")

    def hp(key, default=0):
        v = meta.get(f"{arch}.{key}", default)
        return v if not isinstance(v, list) else (max(v) if v else default)

    experts = {}          # layer -> bytes of all its routed experts
    n_expert = hp("expert_count")
    dense = embd = 0
    attn_layers = set()
    for name, t in tensors.items():
        m = EXPERT_RE.match(name)
        if m and len(t["dims"]) == 3:
            layer = int(m.group(1))
            experts[layer] = experts.get(layer, 0) + t["bytes"]
            n_expert = n_expert or t["dims"][2]
            continue
        if name.startswith("token_embd"):
            embd += t["bytes"]     # a row lookup: stays in RAM
            continue
        dense += t["bytes"]
        m = re.match(r"blk\.(\d+)\.attn_(k|kv_a_mqa)\.weight$", name)
        if m:
            attn_layers.add(int(m.group(1)))

    n_layer = hp("block_count")
    head_kv = hp("attention.head_count_kv") or hp("attention.head_count")
    key_len = hp("attention.key_length") or (hp("embedding_length") // max(1, hp("attention.head_count", 1)))
    val_len = hp("attention.value_length") or key_len
    n_attn = len(attn_layers) or n_layer
    expert_total = sum(experts.values())
    per_expert = {l: b // n_expert for l, b in experts.items()} if n_expert else {}
    return {
        "path": path,
        "arch": arch,
        "name": meta.get("general.name", ""),
        "file_bytes": sum(os.path.getsize(p) for p in shards(path)),
        "n_layer": n_layer,
        "n_expert": n_expert,
        "n_expert_used": hp("expert_used_count"),
        "moe_layers": len(experts),
        "expert_bytes": expert_total,
        "expert_bytes_per_expert": (expert_total // (len(experts) * n_expert)) if experts and n_expert else 0,
        "expert_bytes_per_layer_expert": per_expert,
        "dense_bytes": dense,
        "embedding_bytes": embd,
        "attention_layers": n_attn,
        "head_count_kv": head_kv,
        "key_length": key_len,
        "value_length": val_len,
        "context_length": hp("context_length"),
    }


def kv_bytes(info, ctx, bytes_per_value=1.0625):
    """KV cache of the attention layers at `ctx` tokens (q8_0 by default: 34 bytes per 32 values)."""
    per_token = info["attention_layers"] * info["head_count_kv"] * (info["key_length"] + info["value_length"])
    return int(per_token * ctx * bytes_per_value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    info = model_info(args.model)
    if args.json:
        info = dict(info, expert_bytes_per_layer_expert={str(k): v for k, v in info["expert_bytes_per_layer_expert"].items()})
        print(json.dumps(info, indent=1))
        return
    gib = 1 << 30
    print(f"{os.path.basename(args.model)}: {info['arch']} '{info['name']}', {info['file_bytes'] / gib:.1f} GiB")
    print(f"  layers {info['n_layer']} ({info['moe_layers']} MoE, {info['attention_layers']} with a KV cache), "
          f"experts {info['n_expert']} per layer, {info['n_expert_used']} used per token")
    print(f"  routed experts {info['expert_bytes'] / gib:.1f} GiB ({info['expert_bytes_per_expert'] / (1 << 20):.2f} MiB each), "
          f"dense {info['dense_bytes'] / gib:.2f} GiB, token embedding {info['embedding_bytes'] / gib:.2f} GiB")
    if info["n_expert_used"] and info["moe_layers"]:
        tok = info["n_expert_used"] * info["moe_layers"] * info["expert_bytes_per_expert"]
        print(f"  per token: {tok / (1 << 20):.0f} MiB of experts + {info['dense_bytes'] / (1 << 20):.0f} MiB dense")
    print(f"  KV cache (q8_0): {kv_bytes(info, 32768) / gib:.2f} GiB at 32K context")


if __name__ == "__main__":
    main()
