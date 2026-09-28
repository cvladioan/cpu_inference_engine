#!/usr/bin/env python3
"""Write a small random-weight llama GGUF for testing the deploy scripts.

The model produces gibberish; it only exists so serve.sh, smoke_test.py and
bench.sh can be exercised on any machine without downloading ~160 GB.
The vocabulary is ASCII-only so every generated token is valid UTF-8.
Standard library only.

    python3 deploy/test/make_tiny_gguf.py /tmp/tiny.gguf
    python3 deploy/test/make_tiny_gguf.py --size-gb 20 /data/moe20.gguf

--size-gb writes a Mixture-of-Experts model of about that size instead. Make
it larger than RAM to exercise SSD expert streaming (EXPERT_STREAMING).
Expert weights repeat one random block so tens of GB are written quickly.
"""

import argparse
import math
import random
import struct
from array import array

ALIGN = 32
CTX = 4096

# GGUF value types
U32, I32, F32, BOOL, STR, ARR = 4, 5, 6, 7, 8, 9
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


class Repeat:
    """`nbytes` of float32 data made by repeating one random block."""

    def __init__(self, block: array, n_floats: int):
        self.block = block.tobytes()
        self.nbytes = n_floats * 4

    def write(self, f):
        left = self.nbytes
        while left:
            chunk = self.block[: min(left, len(self.block))]
            f.write(chunk)
            left -= len(chunk)


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key, vtype, value):
    out = gguf_str(key) + struct.pack("<I", vtype)
    if vtype == STR:
        return out + gguf_str(value)
    if vtype == ARR:
        etype, items = value
        out += struct.pack("<IQ", etype, len(items))
        if etype == STR:
            return out + b"".join(gguf_str(x) for x in items)
        fmt = {I32: "<i", F32: "<f"}[etype]
        return out + b"".join(struct.pack(fmt, x) for x in items)
    fmt = {U32: "<I", I32: "<i", F32: "<f", BOOL: "<?"}[vtype]
    return out + struct.pack(fmt, value)


def vocab():
    # SentencePiece-style: control tokens, word boundary, printable ASCII, newline byte.
    tokens = ["<unk>", "<s>", "</s>", "▁"] + [chr(c) for c in range(0x21, 0x7F)] + ["<0x0A>"]
    types = [2, 3, 3] + [1] * (len(tokens) - 4) + [6]
    scores = [0.0, 0.0, 0.0] + [-float(i) for i in range(len(tokens) - 3)]
    return tokens, types, scores


def build(size_gb):
    """Return (hyperparameters, tensors). Tensors are (name, ggml dims, data)."""
    rng = random.Random(0)

    def rand(n, scale=0.05):
        return array("f", (rng.gauss(0.0, scale) for _ in range(n)))

    def ones(n):
        return array("f", [1.0] * n)

    if size_gb:
        # 64 experts, 4 active: each layer holds 3 x 64 x 1024 x 1024 fp32 = 805 MB of experts.
        hp = dict(n_embd=1024, n_ff=1024, n_head=8, n_expert=64, n_expert_used=4)
        per_layer = 3 * hp["n_expert"] * hp["n_embd"] * hp["n_ff"] * 4
        hp["n_layer"] = max(1, math.ceil(size_gb * 1e9 / per_layer))
    else:
        hp = dict(n_embd=256, n_ff=512, n_head=4, n_expert=0, n_expert_used=0, n_layer=2)
    e, ff, nx = hp["n_embd"], hp["n_ff"], hp["n_expert"]
    tokens, _, _ = vocab()
    n_vocab = len(tokens)
    block = rand(1 << 18)  # 1 MB reused for all expert weights

    tensors = [("token_embd.weight", [e, n_vocab], rand(e * n_vocab, 0.5))]
    for i in range(hp["n_layer"]):
        p = f"blk.{i}."
        tensors += [
            (p + "attn_norm.weight", [e], ones(e)),
            (p + "attn_q.weight", [e, e], rand(e * e)),
            (p + "attn_k.weight", [e, e], rand(e * e)),
            (p + "attn_v.weight", [e, e], rand(e * e)),
            (p + "attn_output.weight", [e, e], rand(e * e)),
            (p + "ffn_norm.weight", [e], ones(e)),
        ]
        if nx:
            tensors += [
                (p + "ffn_gate_inp.weight", [e, nx], rand(e * nx, 0.5)),
                (p + "ffn_gate_exps.weight", [e, ff, nx], Repeat(block, e * ff * nx)),
                (p + "ffn_up_exps.weight", [e, ff, nx], Repeat(block, e * ff * nx)),
                (p + "ffn_down_exps.weight", [ff, e, nx], Repeat(block, ff * e * nx)),
            ]
        else:
            tensors += [
                (p + "ffn_gate.weight", [e, ff], rand(e * ff)),
                (p + "ffn_up.weight", [e, ff], rand(e * ff)),
                (p + "ffn_down.weight", [ff, e], rand(ff * e)),
            ]
    tensors += [
        ("output_norm.weight", [e], ones(e)),
        ("output.weight", [e, n_vocab], rand(e * n_vocab, 0.5)),
    ]
    return hp, tensors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--size-gb", type=float, default=0, help="write a MoE model of about this size")
    args = ap.parse_args()

    hp, tensors = build(args.size_gb)
    tokens, types, scores = vocab()
    kvs = [
        kv("general.architecture", STR, "llama"),
        kv("general.name", STR, "tiny-random-test"),
        kv("general.alignment", U32, ALIGN),
        kv("llama.context_length", U32, CTX),
        kv("llama.embedding_length", U32, hp["n_embd"]),
        kv("llama.block_count", U32, hp["n_layer"]),
        kv("llama.feed_forward_length", U32, hp["n_ff"]),
        kv("llama.attention.head_count", U32, hp["n_head"]),
        kv("llama.attention.head_count_kv", U32, hp["n_head"]),
        kv("llama.rope.dimension_count", U32, hp["n_embd"] // hp["n_head"]),
        kv("llama.attention.layer_norm_rms_epsilon", F32, 1e-5),
        kv("tokenizer.ggml.model", STR, "llama"),
        kv("tokenizer.ggml.tokens", ARR, (STR, tokens)),
        kv("tokenizer.ggml.scores", ARR, (F32, scores)),
        kv("tokenizer.ggml.token_type", ARR, (I32, types)),
        kv("tokenizer.ggml.unknown_token_id", U32, 0),
        kv("tokenizer.ggml.bos_token_id", U32, 1),
        kv("tokenizer.ggml.eos_token_id", U32, 2),
        kv("tokenizer.chat_template", STR, CHAT_TEMPLATE),
    ]
    if hp["n_expert"]:
        kvs += [
            kv("llama.expert_count", U32, hp["n_expert"]),
            kv("llama.expert_used_count", U32, hp["n_expert_used"]),
        ]

    def nbytes(data):
        return data.nbytes if isinstance(data, Repeat) else len(data) * 4

    infos, offset = [], 0
    for name, dims, data in tensors:
        infos.append(gguf_str(name) + struct.pack("<I", len(dims)) +
                     b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 0, offset))  # type 0 = F32
        offset += -(-nbytes(data) // ALIGN) * ALIGN

    with open(args.path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
        f.write(b"".join(kvs) + b"".join(infos))
        f.write(b"\0" * (-f.tell() % ALIGN))
        for _, _, data in tensors:
            if isinstance(data, Repeat):
                data.write(f)
            else:
                f.write(data.tobytes())
            f.write(b"\0" * (-f.tell() % ALIGN))
        size = f.tell()
    kind = f"MoE {hp['n_expert']} experts, {hp['n_layer']} layers" if hp["n_expert"] else "dense"
    print(f"wrote {args.path}: {kind}, {len(tensors)} tensors, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
