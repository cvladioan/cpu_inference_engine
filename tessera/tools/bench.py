#!/usr/bin/env python3
"""Measure Tessera on this PC: decode and prompt speed of each configuration, and whether they answer the same.

Each mode starts scripts/serve.sh with some settings, warms up, then for each prompt sends the prompt first and
measures the decode of the next --tokens tokens on its own (greedy). Greedy outputs of all modes are compared with
the first mode's: the GPU/CPU split only changes the order of a sum, so the text should match (tiny probability
differences are normal rounding).

    python3 tessera/tools/bench.py                          # modes: cpu (every expert on the CPU), hot (tiers)
    python3 tessera/tools/bench.py --modes hot --threads 6,8,10
    python3 tessera/tools/bench.py --modes cpu,hot --tokens 128 --out results.md

Standard library only.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE = os.path.join(HERE, "..", "scripts", "serve.sh")
PROMPTS = [
    "Explain to a new engineer why large language model decoding on a CPU is limited by memory bandwidth.",
    "Write a Python function that parses a CSV file of orders and returns the total revenue per customer.",
    "List the planets of the solar system with one interesting fact about each.",
    "Write a haiku about autumn, then explain its imagery.",
]


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def complete(base, prompt, n):
    return post(base + "/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1,
                                       "cache_prompt": True, "ignore_eos": True})


def run_mode(name, env_extra, serve_args, port, tokens, log_dir):
    env = dict(os.environ, HOST="127.0.0.1", PORT=str(port), API_KEY="", **env_extra)
    log_path = os.path.join(log_dir, f"bench-{name}.log")
    with open(log_path, "w") as lf:
        srv = subprocess.Popen([SERVE, *serve_args], env=env, stdout=lf, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    try:
        while True:
            try:
                urllib.request.urlopen(base + "/health", timeout=5)
                break
            except OSError:
                if srv.poll() is not None:
                    sys.exit(f"{name}: the server exited, see {log_path}")
                if time.time() - t0 > 3600:
                    sys.exit(f"{name}: not ready after an hour, see {log_path}")
                time.sleep(2)
        load_s = time.time() - t0
        complete(base, "Hello! Please introduce yourself.", 16)   # warm-up
        texts, dec_n, dec_ms, pp_n, pp_ms = [], 0, 0.0, 0, 0.0
        for p in PROMPTS:
            try:
                r = complete(base, p, 1)             # the prompt, so the next request measures decode alone
            except urllib.error.HTTPError as e:
                print(f"  {name}: prompt rejected ({e.code}), skipped", flush=True)
                continue
            t = r.get("timings", {})
            pp_n += t.get("prompt_n", 0)
            pp_ms += t.get("prompt_ms", 0.0)
            r = complete(base, p, tokens)
            t = r.get("timings", {})
            dec_n += t.get("predicted_n", 0)
            dec_ms += t.get("predicted_ms", 0.0)
            texts.append(r.get("content", ""))
            print(f"  {name}: {t.get('predicted_per_second', 0):.1f} tok/s", flush=True)
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            srv.kill()
    hot_line = ""
    for line in open(log_path, errors="replace"):
        if "hot experts (" in line or "hot experts disabled" in line:
            hot_line = line.strip().split(": ", 1)[-1]
    return {"mode": name, "decode_tok_s": dec_n / dec_ms * 1000 if dec_ms else 0.0,
            "prompt_tok_s": pp_n / pp_ms * 1000 if pp_ms else 0.0, "load_s": load_s, "texts": texts, "hot": hot_line}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="cpu,hot", help="cpu (no hot experts) and/or hot (default: cpu,hot)")
    ap.add_argument("--threads", default="", help="comma-separated THREADS values to try (default: tessera.env)")
    ap.add_argument("--tokens", type=int, default=96, help="greedy tokens per prompt (default 96)")
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--out", help="also write the table (markdown) to this file")
    ap.add_argument("--log-dir", default=os.environ.get("TMPDIR", "/tmp"))
    args = ap.parse_args()

    runs = []
    threads = [t for t in args.threads.split(",") if t] or [None]
    for mode in [m for m in args.modes.split(",") if m]:
        if mode not in ("cpu", "hot"):
            sys.exit(f"unknown mode {mode}")
        for th in threads:
            name = mode + (f"-t{th}" if th else "")
            env = {"THREADS": th} if th else {}
            print(f"== {name}", flush=True)
            runs.append(run_mode(name, env, ["--no-hot"] if mode == "cpu" else [], args.port, args.tokens, args.log_dir))

    ref = runs[0]
    lines = ["| Mode | Decode tok/s | Prompt tok/s | Load s | Same text as first | Hot experts |",
             "|---|---|---|---|---|---|"]
    for r in runs:
        same = "reference" if r is ref else ("yes" if r["texts"] == ref["texts"] else "**no**")
        lines.append(f"| {r['mode']} | {r['decode_tok_s']:.1f} | {r['prompt_tok_s']:.0f} | {r['load_s']:.0f} | {same} | "
                     f"{r['hot'] or '-'} |")
    table = "\n".join(lines)
    print("\n" + table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
