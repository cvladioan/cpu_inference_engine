#!/usr/bin/env python3
"""Print deterministic (greedy) completions and their top-5 token probabilities as JSON.

Used to check that an engine change does not alter results: run it against two
servers (e.g. with and without --expert-cache) and compare the JSON.

    python3 deploy/test/greedy_outputs.py http://127.0.0.1:8080 > out.json
    python3 deploy/test/greedy_outputs.py --compare a.json b.json
"""

import json
import sys
import urllib.request

PROMPTS = ["The quick brown fox", "def parse(x):\n    return", "Numbers: 1, 2, 3, 4,"]


def collect(url):
    out = []
    for prompt in PROMPTS:
        body = {"prompt": prompt, "n_predict": 48, "temperature": 0, "top_k": 1, "n_probs": 5,
                "cache_prompt": False, "ignore_eos": True}
        req = urllib.request.Request(url.rstrip("/") + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        probs = [[(p.get("token") or p.get("tok_str"), round(p.get("prob", p.get("logprob", 0)), 6))
                  for p in (step.get("probs") or step.get("top_probs") or step.get("top_logprobs") or [])]
                 for step in r.get("completion_probabilities", [])]
        out.append({"content": r["content"], "probs": probs})
    return out


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        a, b = (json.load(open(p)) for p in sys.argv[2:])
        same_text = all(x["content"] == y["content"] for x, y in zip(a, b))
        same_probs = all(x["probs"] == y["probs"] for x, y in zip(a, b))
        print(f"identical text: {same_text}, identical top-5 probabilities: {same_probs}")
        sys.exit(0 if same_text and same_probs else 1)
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    json.dump(collect(sys.argv[1]), sys.stdout)


if __name__ == "__main__":
    main()
