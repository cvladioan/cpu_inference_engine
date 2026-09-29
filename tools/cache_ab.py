#!/usr/bin/env python3
"""A/B benchmark of the explicit expert cache against the OS page cache on a
real model (docs/EXPERT_CACHE.md).

For each mode it empties the page cache, starts deploy/serve.sh, warms up,
then generates greedy completions for a fixed prompt set. It records decode
and prompt speed, disk reads per generated token, the expert cache hit rate,
and the outputs, which must be identical across modes.

Modes:
  page       experts streamed through the OS page cache (--defer-experts --prefetch-experts)
  cache      expert cache, budget sized by deploy/ (EXPERT_CACHE_MIB=auto)
  cache:N    expert cache with an N MiB budget
  ram        whole model in RAM (EXPERT_STREAMING=off); only for models that fit

    python3 tools/cache_ab.py                          # model and settings from deploy/config.env
    python3 tools/cache_ab.py --model ~/m.gguf --modes page,cache:4000,cache --leave 6

--leave N holds RAM (tools/ram_limit.py) so only N GiB stay available: this
emulates a smaller machine and forces a model that fits to stream. Emptying the
page cache needs root or `sudo -n tee /proc/sys/vm/drop_caches`; without it the
modes are not comparable and the summary says so.

Writes summary.md, runs.json, outputs and server logs to --out (default
$INSTALL_DIR/results/cache-ab-<host>-<model>-<time>). Standard library only.
"""

import argparse
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"

# Varied content on purpose: routing locality differs between prose, code and lists.
PROMPTS = [
    "Explain to a new engineer why large language model decoding on a CPU is limited by memory bandwidth.",
    "Write a Python function that parses a CSV file of orders and returns the total revenue per customer.",
    "List the planets of the solar system with one interesting fact about each.",
    "用三句话介绍一下长城的历史。",
]

FINAL_RE = re.compile(r"expert cache \(final\): ([\d.]+)% hit rate \((\d+) hits, (\d+) misses\), "
                      r"([\d.]+) GiB read, (\d+) evictions, ([\d.]+) / ([\d.]+) GiB resident")
BUDGET_RE = re.compile(r"expert cache enabled, budget ([\d.]+) GiB")
LOWERED_RE = re.compile(r"budget lowered from [\d.]+ to ([\d.]+) GiB")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def deploy_config():
    """INSTALL_DIR, IK_BIN and the resolved model path, from deploy/lib.sh."""
    script = 'source "$1/lib.sh"; load_config; echo "$INSTALL_DIR"; echo "$IK_BIN"; resolve_model || true'
    out = subprocess.run(["bash", "-c", script, "_", str(DEPLOY)], capture_output=True, text=True).stdout
    lines = out.splitlines() + ["", "", ""]
    return lines[0], lines[1], lines[2]


def meminfo(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) // 1024  # MiB
    return 0


def disk_read_kib():
    with open("/proc/vmstat") as f:
        for line in f:
            if line.startswith("pgpgin "):
                return int(line.split()[1])
    return 0


def drop_caches():
    subprocess.run(["sync"])
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except OSError:
        pass
    r = subprocess.run(["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"], input="3\n", text=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 and sys.stdin.isatty():
        log("emptying the page cache needs sudo:")
        r = subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
    return r.returncode == 0


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def complete(base, prompt, n, timeout):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "n_probs": 5,
            "cache_prompt": True, "ignore_eos": True}
    return post(base + "/completion", body, timeout)


def top_probs(r):
    return [[(p.get("token") or p.get("tok_str"), round(p.get("prob", p.get("logprob", 0)), 6))
             for p in (step.get("probs") or step.get("top_probs") or step.get("top_logprobs") or [])]
            for step in r.get("completion_probabilities", [])]


def machine_info(model):
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "?")
    try:
        version = open("/proc/version").read()
    except OSError:
        version = ""
    return {
        "host": socket.gethostname(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "mem_total_mib": meminfo("MemTotal"),
        "mem_available_mib": meminfo("MemAvailable"),
        "swap_total_mib": meminfo("SwapTotal"),
        "wsl": "microsoft" in version.lower(),
        "kernel": platform.release(),
        "model": str(model),
        "model_gib": round(model_size(model) / 2**30, 1),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }


def model_size(model):
    """Total size of a model, including the other shards of a split GGUF."""
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", model.name)
    if not m:
        return model.stat().st_size
    return sum(p.stat().st_size for p in model.parent.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf"))


def mode_env(mode):
    if mode == "page":
        return {"EXPERT_STREAMING": "on", "EXPERT_CACHE_MIB": "0"}
    if mode == "ram":
        return {"EXPERT_STREAMING": "off", "EXPERT_CACHE_MIB": "0"}
    if mode == "cache":
        return {"EXPERT_STREAMING": "on", "EXPERT_CACHE_MIB": "auto"}
    if mode.startswith("cache:") and mode[6:].isdigit():
        return {"EXPERT_STREAMING": "on", "EXPERT_CACHE_MIB": mode[6:]}
    sys.exit(f"unknown mode: {mode} (page, cache, cache:N, ram)")


def run_mode(args, mode, model, out_dir, hog):
    tag = mode.replace(":", "-")
    run = {"mode": mode, "valid": True, "notes": []}
    run["caches_dropped"] = drop_caches() if not args.no_drop else False
    if not run["caches_dropped"]:
        run["notes"].append("page cache not emptied")

    env = dict(os.environ, MODEL_FILE=str(model), HOST="127.0.0.1", PORT=str(args.port), API_KEY="",
               API_KEY_FILE="", PARALLEL="1", CTX_PER_SLOT=str(args.ctx), CACHE_RAM_MIB="0", SPEC_TYPE="",
               NUMA_MODE=os.environ.get("NUMA_MODE", "none"), **mode_env(mode))
    if args.threads:
        env["THREADS"] = str(args.threads)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    plan = subprocess.run([str(DEPLOY / "serve.sh"), "--plan"], env=env, capture_output=True, text=True).stdout
    run["plan"] = dict(line.split("=", 1) for line in plan.splitlines() if "=" in line)
    if mode.startswith("cache") and run["plan"].get("EXPERT_CACHE_MIB", "0") == "0":
        run["valid"] = False
        run["notes"].append("expert cache not active (unpatched build or no RAM for it); see serve log")

    log_path = out_dir / f"server-{tag}.log"
    base = f"http://127.0.0.1:{args.port}"
    log(f"{mode}: starting server (plan: {run['plan']})")
    t0 = time.time()
    with open(log_path, "w") as log_file:
        server = subprocess.Popen([str(DEPLOY / "serve.sh")], env=env, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        while True:
            try:
                urllib.request.urlopen(base + "/health", timeout=5)
                break
            except OSError:
                if server.poll() is not None:
                    sys.exit(f"{mode}: server exited, see {log_path}")
                if time.time() - t0 > args.load_timeout:
                    sys.exit(f"{mode}: server not ready after {args.load_timeout}s, see {log_path}")
                time.sleep(2)
        run["load_s"] = round(time.time() - t0, 1)

        log(f"{mode}: warm-up ({args.warmup} tokens)")
        complete(base, "Hello! Please introduce yourself.", args.warmup, args.request_timeout)

        outputs, dec_n, dec_ms, pp_n, pp_ms, disk_kib = [], 0, 0.0, 0, 0.0, 0
        for i, prompt in enumerate(PROMPTS):
            # Prompt first, so the measured request is decode only (its prompt comes from the KV cache).
            try:
                r = complete(base, prompt, 1, args.request_timeout)
            except urllib.error.HTTPError as e:
                # Deterministic (e.g. a test model's ASCII-only vocabulary), so every mode skips it.
                log(f"{mode}: prompt {i + 1} rejected ({e.code}: {e.read().decode(errors='replace')[:200]}), skipped")
                continue
            t = r.get("timings", {})
            pp_n += t.get("prompt_n", 0)
            pp_ms += t.get("prompt_ms", 0.0)
            before = disk_read_kib()
            r = complete(base, prompt, args.tokens, args.request_timeout)
            disk_kib += disk_read_kib() - before
            t = r.get("timings", {})
            dec_n += t.get("predicted_n", 0)
            dec_ms += t.get("predicted_ms", 0.0)
            outputs.append({"prompt": prompt, "content": r.get("content", ""), "probs": top_probs(r)})
            tps = t.get("predicted_n", 0) / max(t.get("predicted_ms", 1), 1) * 1000
            log(f"{mode}: prompt {i + 1}/{len(PROMPTS)}: {tps:.2f} tok/s")
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=180)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    if hog is not None and hog.poll() is not None:
        run["valid"] = False
        run["notes"].append("RAM holder died (out of memory?)")

    text = log_path.read_text(errors="replace")
    run["decode_tok_s"] = round(dec_n / dec_ms * 1000, 3) if dec_ms else None
    run["prompt_tok_s"] = round(pp_n / pp_ms * 1000, 2) if pp_ms else None
    run["tokens"] = dec_n
    run["disk_mib_per_token"] = round(disk_kib / 1024 / dec_n, 1) if dec_n else None
    if m := BUDGET_RE.search(text):
        run["cache_budget_gib"] = float(m.group(1))
    if m := LOWERED_RE.search(text):  # the engine fitted the budget to available memory
        run["requested_budget_gib"] = run.get("cache_budget_gib")
        run["cache_budget_gib"] = float(m.group(1))
    if finals := FINAL_RE.findall(text):
        hit, hits, misses, read_gib, evictions, resident, budget = finals[-1]
        run.update(hit_rate=float(hit), hits=int(hits), misses=int(misses), cache_read_gib=float(read_gib),
                   evictions=int(evictions))
    (out_dir / f"outputs-{tag}.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=1))
    run["outputs"] = outputs
    log(f"{mode}: {run['decode_tok_s']} tok/s decode, {run['disk_mib_per_token']} MiB/token from disk"
        + (f", {run['hit_rate']}% hit rate" if "hit_rate" in run else ""))
    return run


def summarize(info, runs):
    ref = runs[0]
    lines = [
        f"# Expert cache A/B: {Path(info['model']).name}",
        "",
        f"- Machine: {info['cpu']}, {info['logical_cpus']} logical CPUs, "
        f"{info['mem_total_mib'] / 1024:.1f} GiB RAM ({info['mem_available_mib'] / 1024:.1f} GiB available at start), "
        f"swap {info['swap_total_mib'] / 1024:.1f} GiB{', WSL2' if info['wsl'] else ''}",
        f"- Model: {info['model_gib']} GiB; {info['date']}; {info.get('leave', 'no RAM limit')}",
        f"- {len(ref['outputs'])} prompts x {ref['tokens'] // max(len(ref['outputs']), 1)} greedy tokens per mode, "
        "decode only (each prompt is processed in a separate request first)",
        "",
        "| Mode | Cache budget | Decode tok/s | Prompt tok/s | Disk MiB/token | Hit rate | Outputs vs first mode | Valid |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        same_text = all(a["content"] == b["content"] for a, b in zip(r["outputs"], ref["outputs"]))
        same_probs = all(a["probs"] == b["probs"] for a, b in zip(r["outputs"], ref["outputs"]))
        r["identical_to_first"] = same_text and same_probs
        same = "reference" if r is ref else ("identical" if r["identical_to_first"]
                                             else "**text differs**" if not same_text else "**probs differ**")
        budget = f"{r['cache_budget_gib']:.2f} GiB" if "cache_budget_gib" in r else "-"
        if "requested_budget_gib" in r:
            budget += f" (lowered from {r['requested_budget_gib']:.2f})"
        hit = f"{r['hit_rate']:.1f}%" if "hit_rate" in r else "-"
        valid = "yes" if r["valid"] and r["caches_dropped"] else "no: " + "; ".join(r["notes"])
        lines.append(f"| {r['mode']} | {budget} | {r['decode_tok_s']} | {r['prompt_tok_s']} | "
                     f"{r['disk_mib_per_token']} | {hit} | {same} | {valid} |")
    base = next((r for r in runs if r["mode"] == "page" and r["decode_tok_s"]), None)
    if base:
        lines.append("")
        for r in runs:
            if r is not base and r["decode_tok_s"]:
                a, b = base["disk_mib_per_token"] or 0, r["disk_mib_per_token"] or 0
                disk = (f"{a / b:.1f}x less disk traffic" if a >= 1 and b >= 0.1
                        else f"disk {a} vs {b} MiB/token")
                lines.append(f"- {r['mode']}: {r['decode_tok_s'] / base['decode_tok_s']:.2f}x the decode speed "
                             f"of page, {disk}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="GGUF file (first shard of a split model); default: deploy/config.env")
    ap.add_argument("--modes", default="page,cache", help="comma-separated: page, cache, cache:N, ram (default page,cache)")
    ap.add_argument("--tokens", type=int, default=64, help="greedy tokens per prompt (default 64)")
    ap.add_argument("--warmup", type=int, default=16, help="warm-up tokens before measuring (default 16)")
    ap.add_argument("--leave", type=float, help="hold RAM so only this many GiB stay available")
    ap.add_argument("--threads", type=int, help="override THREADS")
    ap.add_argument("--ctx", type=int, default=4096, help="context size (default 4096)")
    ap.add_argument("--port", type=int, default=18700)
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the server (repeatable)")
    ap.add_argument("--out", help="results directory")
    ap.add_argument("--no-drop", action="store_true", help="do not empty the page cache between modes")
    ap.add_argument("--load-timeout", type=int, default=3600, help="seconds to wait for the server (default 3600)")
    ap.add_argument("--request-timeout", type=int, default=7200, help="seconds per request (default 7200)")
    args = ap.parse_args()

    install_dir, _, cfg_model = deploy_config()
    model = Path(os.path.expanduser(args.model or cfg_model))
    if not model.is_file():
        sys.exit(f"model not found: {model!s} (pass --model or set MODEL_FILE/MODEL_DIR in deploy/config.env)")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        mode_env(m)

    info = machine_info(model)
    if info["swap_total_mib"] and any(m.startswith("cache") for m in modes):
        log("warning: swap is on; the kernel may swap cached experts out (sudo swapoff -a)")
    out_dir = Path(args.out or Path(install_dir or ".") / "results" /
                   f"cache-ab-{info['host']}-{model.stem}-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"model {model.name} ({info['model_gib']} GiB), modes {modes}, results in {out_dir}")

    hog = None
    if args.leave is not None:
        hog_log = out_dir / "ram_limit.log"
        hog = subprocess.Popen([sys.executable, str(ROOT / "tools" / "ram_limit.py"), "--leave", str(args.leave)],
                               stdout=open(hog_log, "w"), stderr=subprocess.STDOUT)
        while "holding" not in hog_log.read_text():
            if hog.poll() is not None:
                sys.exit(f"ram_limit.py exited: {hog_log.read_text()}")
            time.sleep(1)
        info["leave"] = f"RAM limited to {args.leave} GiB available ({hog_log.read_text().strip()})"
        log(info["leave"])

    runs = []
    try:
        for m in modes:
            runs.append(run_mode(args, m, model, out_dir, hog))
    finally:
        if hog is not None:
            hog.terminate()
            hog.wait()

    summary = summarize(info, runs)
    (out_dir / "summary.md").write_text(summary)
    (out_dir / "runs.json").write_text(json.dumps({"machine": info, "runs": runs}, ensure_ascii=False, indent=1))
    print("\n" + summary)
    log(f"saved {out_dir}/summary.md")
    sys.exit(0 if all(r.get("identical_to_first") for r in runs) else 1)


if __name__ == "__main__":
    main()
