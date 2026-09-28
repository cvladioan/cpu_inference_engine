#!/usr/bin/env python3
"""Write a tiny random-weight llama GGUF for testing the deploy scripts.

The model produces gibberish; it only exists so serve.sh, smoke_test.py and
bench.sh can be exercised on any machine without downloading ~160 GB.
The vocabulary is ASCII-only so every generated token is valid UTF-8.
Standard library only.

    python3 deploy/test/make_tiny_gguf.py /tmp/tiny.gguf
"""

import random
import struct
import sys
from array import array

ALIGN = 32
N_EMBD, N_FF, N_LAYER, N_HEAD = 256, 512, 2, 4
CTX = 4096

# GGUF value types
U32, I32, F32, BOOL, STR, ARR = 4, 5, 6, 7, 8, 9
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


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


def main(path):
    rng = random.Random(0)
    tokens, types, scores = vocab()
    n_vocab = len(tokens)

    def rand(n, scale=0.05):
        return array("f", (rng.gauss(0.0, scale) for _ in range(n)))

    def ones(n):
        return array("f", [1.0] * n)

    # (name, ggml dims ne0..ne1, data). ne0 is the contiguous dimension.
    tensors = [("token_embd.weight", [N_EMBD, n_vocab], rand(N_EMBD * n_vocab, 0.5))]
    for i in range(N_LAYER):
        p = f"blk.{i}."
        tensors += [
            (p + "attn_norm.weight", [N_EMBD], ones(N_EMBD)),
            (p + "attn_q.weight", [N_EMBD, N_EMBD], rand(N_EMBD * N_EMBD)),
            (p + "attn_k.weight", [N_EMBD, N_EMBD], rand(N_EMBD * N_EMBD)),
            (p + "attn_v.weight", [N_EMBD, N_EMBD], rand(N_EMBD * N_EMBD)),
            (p + "attn_output.weight", [N_EMBD, N_EMBD], rand(N_EMBD * N_EMBD)),
            (p + "ffn_norm.weight", [N_EMBD], ones(N_EMBD)),
            (p + "ffn_gate.weight", [N_EMBD, N_FF], rand(N_EMBD * N_FF)),
            (p + "ffn_up.weight", [N_EMBD, N_FF], rand(N_EMBD * N_FF)),
            (p + "ffn_down.weight", [N_FF, N_EMBD], rand(N_FF * N_EMBD)),
        ]
    tensors += [
        ("output_norm.weight", [N_EMBD], ones(N_EMBD)),
        ("output.weight", [N_EMBD, n_vocab], rand(N_EMBD * n_vocab, 0.5)),
    ]

    kvs = [
        kv("general.architecture", STR, "llama"),
        kv("general.name", STR, "tiny-random-test"),
        kv("general.alignment", U32, ALIGN),
        kv("llama.context_length", U32, CTX),
        kv("llama.embedding_length", U32, N_EMBD),
        kv("llama.block_count", U32, N_LAYER),
        kv("llama.feed_forward_length", U32, N_FF),
        kv("llama.attention.head_count", U32, N_HEAD),
        kv("llama.attention.head_count_kv", U32, N_HEAD),
        kv("llama.rope.dimension_count", U32, N_EMBD // N_HEAD),
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

    infos, offset = [], 0
    for name, dims, data in tensors:
        infos.append(gguf_str(name) + struct.pack("<I", len(dims)) +
                     b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 0, offset))  # type 0 = F32
        offset += -(-len(data) * 4 // ALIGN) * ALIGN

    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
        f.write(b"".join(kvs) + b"".join(infos))
        f.write(b"\0" * (-f.tell() % ALIGN))
        for _, _, data in tensors:
            f.write(data.tobytes())
            f.write(b"\0" * (-f.tell() % ALIGN))
    print(f"wrote {path}: {len(tensors)} tensors, vocab {n_vocab}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
