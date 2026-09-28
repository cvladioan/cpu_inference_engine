#!/usr/bin/env python3
"""Check a running server end to end and measure what users see.

Sends streaming chat requests to the OpenAI-compatible API and reports time to
first token (TTFT) and generation speed per request and in aggregate.

    python3 deploy/smoke_test.py
    python3 deploy/smoke_test.py --url http://10.0.0.5:8080 --api-key KEY --concurrency 4

Exit code is non-zero if the server is unhealthy or any request fails.
Standard library only.
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = "Explain in about 150 words why large language model decoding on CPUs is memory-bandwidth bound."


def request(url, api_key, body=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout)


def chat_stream(args, prompt, result):
    body = {
        "model": args.model or "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.6,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.fixed_length:
        # llama.cpp extension: keep generating past end-of-sequence.
        body["ignore_eos"] = True
    start = time.perf_counter()
    first = None
    chunks = 0
    n_tokens = None
    server_tps = None
    text = []
    try:
        with request(f"{args.url}/v1/chat/completions", args.api_key, body, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content")
                    if piece:
                        if first is None:
                            first = time.perf_counter()
                        chunks += 1
                        text.append(piece)
                usage = event.get("usage") or {}
                if usage.get("completion_tokens"):
                    n_tokens = usage["completion_tokens"]
                timings = event.get("timings") or {}
                if timings.get("predicted_n"):
                    n_tokens = timings["predicted_n"]
                    server_tps = timings.get("predicted_per_second")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        result["error"] = str(exc)
        return
    end = time.perf_counter()
    n = n_tokens or chunks
    result.update(
        ttft=(first - start) if first else None,
        tokens=n,
        # Generation speed excludes the prompt phase: tokens after the first one.
        tps=server_tps or ((n - 1) / (end - first) if first and n > 1 and end > first else None),
        wall=end - start,
        text="".join(text),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None, help="model name to send (default: first from /v1/models)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--fixed-length", action="store_true",
                    help="generate exactly --max-tokens per request (comparable speed numbers)")
    ap.add_argument("--show-text", action="store_true", help="print the first response")
    args = ap.parse_args()
    args.url = args.url.rstrip("/")

    try:
        with request(f"{args.url}/health", args.api_key) as resp:
            print(f"health: {resp.status} {resp.read().decode()[:200]}")
        with request(f"{args.url}/v1/models", args.api_key) as resp:
            models = [m.get("id") for m in json.load(resp).get("data", [])]
            print(f"models: {models}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"FAIL: server not healthy at {args.url}: {exc}")
        return 1
    if not args.model and models:
        args.model = models[0]

    results = [{} for _ in range(args.concurrency)]
    threads = [threading.Thread(target=chat_stream, args=(args, args.prompt, r)) for r in results]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start

    failed = 0
    print(f"\n{'req':>3} {'TTFT s':>8} {'tokens':>7} {'gen tok/s':>10} {'wall s':>7}")
    for i, r in enumerate(results):
        if "error" in r or not r.get("tokens"):
            failed += 1
            print(f"{i:>3} FAILED: {r.get('error', 'no tokens generated')}")
            continue
        ttft = f"{r['ttft']:.2f}" if r["ttft"] is not None else "-"
        tps = f"{r['tps']:.1f}" if r["tps"] else "-"
        print(f"{i:>3} {ttft:>8} {r['tokens']:>7} {tps:>10} {r['wall']:>7.1f}")
    ok = [r for r in results if r.get("tokens")]
    if ok:
        total = sum(r["tokens"] for r in ok)
        print(f"\naggregate: {total} tokens in {wall:.1f} s = {total / wall:.1f} tok/s across {len(ok)} request(s)")
    if args.show_text and ok:
        print("\n--- response 0 ---\n" + ok[0]["text"])
    if failed:
        print(f"FAIL: {failed} request(s) failed")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
