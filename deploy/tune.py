#!/usr/bin/env python3
"""Find the fastest serving configuration on this machine and check it against a target.

Starts deploy/serve.sh once per candidate configuration, measures real
generation speed through the API on three kinds of text (code, prose,
structured extraction), and ranks the results. The search is staged so that
each stage keeps the best setting of the previous one:

  1. threads          physical cores and a few lower counts
  2. weight repacking -rtr (repack weights at load for faster kernels)
  3. speculation      self-speculation (n-gram), and DSpark if a draft is configured
  4. expert count     -ser (fewer experts per token; changes outputs, opt-in)

    python3 deploy/tune.py --target 20
    python3 deploy/tune.py --target 20 --apply              # write the winner to deploy/config.env
    python3 deploy/tune.py --threads "6 8 10" --quick        # desktops with P/E cores

Stop the production server first: the tuner needs the whole machine.
Standard library only.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import smoke_test  # noqa: E402  (reuses the streaming client)

PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 date string into a datetime, with type hints, "
            "a docstring and error handling. Then write three pytest tests for it.",
    "prose": "Explain to a new engineer, in plain language, how a CPU cache hierarchy works and why memory "
             "bandwidth limits large language model inference on CPUs.",
    "extract": "Convert to JSON with keys name, role, start_year, skills (list of strings): \"Maria Popescu "
               "joined as a data engineer in 2019 and works with Python, Spark, Airflow and PostgreSQL. "
               "Andrei Ionescu joined as a backend developer in 2021 and works with Go, gRPC and Redis.\"",
}
CONFIG_KEYS = ("THREADS", "SPEC_TYPE", "EXTRA_ARGS", "DRAFT_FILE", "DRAFT_REPO", "INSTALL_DIR", "MODEL_ALIAS")


def load_config():
    """Resolved deploy settings, exactly as serve.sh sees them."""
    script = ('source "$1/lib.sh"; load_config; '
              'for v in "${@:2}"; do printf "%s=%s\\0" "$v" "${!v}"; done; '
              'printf "CORES=%s\\0" "$(physical_cores)"; '
              'printf "CORES_NODE0=%s\\0" "$(physical_cores 0)"; '
              'printf "DRAFT=%s\\0" "$(resolve_draft 2>/dev/null)"; '
              'printf "PLAN=%s\\0" "$("$1/serve.sh" --plan 2>&1 | tr "\\n" " ")"')
    out = subprocess.run(["bash", "-c", script, "_", str(DEPLOY), *CONFIG_KEYS],
                         capture_output=True, text=True, check=True).stdout
    return dict(item.split("=", 1) for item in out.split("\0") if "=" in item)


class Candidate:
    def __init__(self, threads, spec="", extra="", note=""):
        self.threads, self.spec, self.extra, self.note = threads, spec, extra.strip(), note
        self.result = None  # filled by measure()

    @property
    def label(self):
        parts = [f"threads={self.threads}"]
        if self.extra:
            parts.append(self.extra)
        if self.spec:
            parts.append(f"spec={self.spec}")
        return " ".join(parts)

    def key(self):
        return (self.threads, self.spec, self.extra)


class Tuner:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        self.port = args.port
        self.url = f"http://127.0.0.1:{self.port}"
        self.base_extra = cfg.get("EXTRA_ARGS", "")
        self.plan = cfg.get("PLAN", "")
        self.node_args = ["--node", "0"] if "NUMA_MODE=per-node" in self.plan else []
        self.results = []
        self.log_dir = Path(args.results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def env_for(self, cand):
        env = dict(os.environ)
        env.update(
            HOST="127.0.0.1", PORT=str(self.port), API_KEY="", API_KEY_FILE="",
            PARALLEL="1", CTX_PER_SLOT=str(self.args.ctx), CACHE_RAM_MIB="1024",
            THREADS=str(cand.threads), SPEC_TYPE=cand.spec,
            EXTRA_ARGS=" ".join(x for x in (self.base_extra, cand.extra) if x),
        )
        return env

    def wait_healthy(self, proc, log_path):
        deadline = time.time() + self.args.load_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                return f"server exited (code {proc.returncode}); see {log_path}"
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as resp:
                    if resp.status == 200:
                        return None
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        return f"not healthy after {self.args.load_timeout} s; see {log_path}"

    def generate(self, prompt, max_tokens):
        ns = SimpleNamespace(url=self.url, api_key=None, model=self.cfg.get("MODEL_ALIAS") or "default",
                             max_tokens=max_tokens, fixed_length=True, timeout=self.args.request_timeout)
        result = {}
        smoke_test.chat_stream(ns, prompt, result)
        return result

    def measure(self, cand):
        for done in self.results:
            if done.key() == cand.key():
                return done.result
        idx = len(self.results)
        log_path = self.log_dir / f"tune-{idx:02d}.log"
        print(f"\n[{idx}] {cand.label}{'  (' + cand.note + ')' if cand.note else ''}", flush=True)
        t0 = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.Popen([str(DEPLOY / "serve.sh"), *self.node_args], env=self.env_for(cand),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            err = self.wait_healthy(proc, log_path)
            load_s = time.time() - t0
            res = {"load_s": round(load_s, 1), "error": err, "per_prompt": {}}
            if not err:
                self.generate("Say hello.", 16)  # warm-up: first-token path and page cache
                for name, prompt in PROMPTS.items():
                    runs = [self.generate(prompt, self.args.max_tokens) for _ in range(self.args.repeat)]
                    bad = [r for r in runs if "error" in r or not r.get("tps")]
                    if bad:
                        res["error"] = f"{name}: {bad[0].get('error', 'no tokens')}"
                        break
                    res["per_prompt"][name] = {
                        "tps": round(sum(r["tps"] for r in runs) / len(runs), 2),
                        "ttft": round(sum(r["ttft"] or 0 for r in runs) / len(runs), 2),
                    }
            if not res["error"]:
                res["tps"] = round(sum(p["tps"] for p in res["per_prompt"].values()) / len(PROMPTS), 2)
                detail = ", ".join(f"{k} {v['tps']:.1f}" for k, v in res["per_prompt"].items())
                print(f"    {res['tps']:.1f} tok/s  ({detail}; loaded in {load_s:.0f} s)", flush=True)
            else:
                print(f"    FAILED: {res['error']}", flush=True)
        finally:
            self.stop(proc)
        cand.result = res
        self.results.append(cand)
        return res

    @staticmethod
    def stop(proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass

    def best(self, cands):
        ok = [c for c in cands if c.result and "tps" in c.result]
        return max(ok, key=lambda c: c.result["tps"]) if ok else None

    def run(self):
        # In per-node mode each instance runs on one NUMA node.
        cores_key = "CORES_NODE0" if self.node_args else "CORES"
        cores = int(self.cfg.get(cores_key) or os.cpu_count() or 4)
        if self.args.threads:
            thread_list = [int(t) for t in self.args.threads.split()]
        elif self.cfg.get("THREADS"):
            thread_list = [int(self.cfg["THREADS"])]
        else:
            thread_list = sorted({cores, max(1, cores * 3 // 4)}, reverse=True)

        # Stage 1: threads, no speculation.
        best = self.best([self.stage(Candidate(t)) for t in thread_list])
        if not best:
            return None
        if self.args.quick:
            return best

        # Stage 2: repack weights at load time.
        best = self.best([best, self.stage(Candidate(best.threads, extra=f"{best.extra} -rtr".strip()))])

        # Stage 3: speculative decoding.
        specs = ["ngram-mod:n_max=16,n_min=2,ngram_size_n=8"]
        if self.cfg.get("DRAFT"):
            specs += ["dspark:n_max=3", "dspark:n_max=5"]
        if self.cfg.get("SPEC_TYPE") and self.cfg["SPEC_TYPE"] not in specs:
            specs.append(self.cfg["SPEC_TYPE"])
        best = self.best([best] + [self.stage(Candidate(best.threads, spec=s, extra=best.extra)) for s in specs])

        # Stage 4: fewer experts per token (changes outputs, so only on request).
        if self.args.allow_expert_reduction:
            cands = [Candidate(best.threads, spec=best.spec, extra=f"{best.extra} -ser {k},1".strip(),
                               note="fewer experts: check answer quality") for k in self.args.ser.split()]
            best = self.best([best] + [self.stage(c) for c in cands])
        return best

    def stage(self, cand):
        self.measure(cand)
        return cand

    def report(self, best):
        rows = sorted((c for c in self.results if c.result), key=lambda c: -c.result.get("tps", -1))
        lines = ["| # | configuration | tok/s | code | prose | extract | TTFT s | load s |",
                 "|---|---|---|---|---|---|---|---|"]
        for c in rows:
            r = c.result
            if "tps" in r:
                pp = r["per_prompt"]
                ttft = sum(p["ttft"] for p in pp.values()) / len(pp)
                lines.append(f"| {self.results.index(c)} | {c.label} | **{r['tps']:.1f}** | "
                             f"{pp['code']['tps']:.1f} | {pp['prose']['tps']:.1f} | {pp['extract']['tps']:.1f} | "
                             f"{ttft:.2f} | {r['load_s']:.0f} |")
            else:
                lines.append(f"| {self.results.index(c)} | {c.label} | failed | | | | | {r['load_s']:.0f} |")
        table = "\n".join(lines)
        print("\n" + table)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = self.log_dir / f"tune-{stamp}"
        out.with_suffix(".md").write_text(f"# Tuning results {stamp}\n\n{self.plan}\n\n{table}\n")
        out.with_suffix(".json").write_text(json.dumps(
            [{"label": c.label, "threads": c.threads, "spec": c.spec, "extra": c.extra, **c.result}
             for c in self.results], indent=2))
        print(f"\nsaved {out}.md and .json (server logs: {self.log_dir}/tune-NN.log)")

        if not best:
            print("\nFAIL: no configuration produced tokens; check the server logs above.")
            return 2
        tps = best.result["tps"]
        print(f"\nbest: {best.label} -> {tps:.1f} tok/s")
        if "-ser" in best.extra:
            print("      uses fewer experts per token: compare answers against the default before using it")
        if self.args.target:
            if tps >= self.args.target:
                print(f"PASS: {tps:.1f} tok/s >= target {self.args.target:g} tok/s")
            else:
                print(f"BELOW TARGET: {tps:.1f} tok/s < {self.args.target:g} tok/s. Generation speed is bound by "
                      "memory bandwidth; `python3 tools/roofline.py --target-tps "
                      f"{self.args.target:g}` shows the bandwidth this target needs.")
        if self.args.apply:
            apply_config(best, self.cfg)
        else:
            print("\nto use it, set in deploy/config.env (or rerun with --apply):")
            for k, v in settings_for(best, self.cfg).items():
                print(f'  {k}="{v}"')
        return 0 if not self.args.target or tps >= self.args.target else 1


def settings_for(cand, cfg):
    extra = " ".join(x for x in (cfg.get("EXTRA_ARGS", ""), cand.extra) if x)
    return {"THREADS": str(cand.threads), "SPEC_TYPE": cand.spec, "EXTRA_ARGS": extra}


def apply_config(cand, cfg):
    path = DEPLOY / "config.env"
    if not path.exists():
        shutil.copy(DEPLOY / "config.env.example", path)
    else:
        shutil.copy(path, path.with_suffix(".env.bak"))
    text = path.read_text()
    for key, value in settings_for(cand, cfg).items():
        line = f'{key}="{value}"'
        pattern = re.compile(rf"^{key}=.*$", re.M)
        text = pattern.sub(line, text, count=1) if pattern.search(text) else text.rstrip("\n") + f"\n{line}\n"
    path.write_text(text)
    print(f"\napplied to {path} (previous version: config.env.bak); restart the server to use it")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=0, help="required generation tok/s (exit code 1 if missed)")
    ap.add_argument("--threads", default="", help='thread counts to try, e.g. "6 8 10" (default: auto)')
    ap.add_argument("--quick", action="store_true", help="only tune the thread count")
    ap.add_argument("--allow-expert-reduction", action="store_true",
                    help="also try -ser (fewer experts per token; faster, but changes outputs)")
    ap.add_argument("--ser", default="5 4", help="expert counts to try with --allow-expert-reduction")
    ap.add_argument("--apply", action="store_true", help="write the best settings to deploy/config.env")
    ap.add_argument("--max-tokens", type=int, default=192, help="tokens generated per measurement")
    ap.add_argument("--repeat", type=int, default=1, help="measurements per prompt")
    ap.add_argument("--ctx", type=int, default=8192, help="context size while tuning")
    ap.add_argument("--port", type=int, default=18999)
    ap.add_argument("--load-timeout", type=int, default=1800, help="seconds to wait for the model to load")
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--results-dir", default="", help="default: $INSTALL_DIR/results")
    args = ap.parse_args()

    cfg = load_config()
    args.results_dir = args.results_dir or os.path.join(cfg.get("INSTALL_DIR") or ".", "results")
    print(f"tuning on {cfg.get('CORES')} physical cores; {cfg.get('PLAN', '').strip()}")
    if cfg.get("DRAFT"):
        print(f"draft model: {cfg['DRAFT']}")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2):
            sys.exit(f"port {args.port} is already in use; pass --port")
    except (urllib.error.URLError, OSError):
        pass

    tuner = Tuner(args, cfg)
    try:
        best = tuner.run()
    except KeyboardInterrupt:
        print("\ninterrupted; reporting what was measured")
        best = tuner.best(tuner.results)
    sys.exit(tuner.report(best))


if __name__ == "__main__":
    main()
