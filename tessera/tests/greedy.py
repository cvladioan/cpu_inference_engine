#!/usr/bin/env python3
"""Greedy completions with top-5 probabilities from a running server, and a tolerant comparison of two runs.

    python3 tessera/tests/greedy.py collect http://127.0.0.1:8080 > a.json
    python3 tessera/tests/greedy.py compare a.json b.json [--tol 1e-3]

`compare` passes when the text is identical, the top-5 tokens are the same at every step, and no probability
differs by more than --tol. The GPU/CPU split adds the experts' outputs in a different order, which moves
probabilities by rounding (~1e-5); a wrong expert changes the tokens.
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = ["The quick brown fox", "def parse(x):\n    return", "Numbers: 1, 2, 3, 4,"]


def collect(url):
    out = []
    for prompt in PROMPTS:
        body = {"prompt": prompt, "n_predict": 48, "temperature": 0, "top_k": 1, "n_probs": 5,
                "cache_prompt": False, "ignore_eos": True}
        req = urllib.request.Request(url.rstrip("/") + "/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        probs = [[(p.get("token") or p.get("tok_str"), p.get("prob", p.get("logprob", 0)))
                  for p in (s.get("probs") or s.get("top_probs") or s.get("top_logprobs") or [])]
                 for s in r.get("completion_probabilities", [])]
        out.append({"content": r["content"], "probs": probs})
    return out


def compare(a, b, tol):
    text = all(x["content"] == y["content"] for x, y in zip(a, b))
    same_tokens, worst = True, 0.0
    for x, y in zip(a, b):
        for sa, sb in zip(x["probs"], y["probs"]):
            if [t for t, _ in sa] != [t for t, _ in sb]:
                same_tokens = False
            for (_, pa), (_, pb) in zip(sa, sb):
                worst = max(worst, abs(pa - pb))
    ok = text and same_tokens and worst <= tol
    print(f"text identical: {text}, top-5 tokens identical: {same_tokens}, max |dprob| {worst:.2e} (tol {tol:g}): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("url")
    k = sub.add_parser("compare")
    k.add_argument("a")
    k.add_argument("b")
    k.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    if args.cmd == "collect":
        json.dump(collect(args.url), sys.stdout)
    else:
        sys.exit(0 if compare(json.load(open(args.a)), json.load(open(args.b)), args.tol) else 1)


if __name__ == "__main__":
    main()
