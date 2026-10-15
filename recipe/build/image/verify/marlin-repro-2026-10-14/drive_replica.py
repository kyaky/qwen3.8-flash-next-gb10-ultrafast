#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive the marlin-replica server like the D-011 flip probe: N sequential
requests sharing one long cached prefix, each with a DISTINCT 24-token
suffix, so every request produces an ntok=24 prefill forward (the same
marlin GEMM1 M=24 / GEMM2 M=240 call context that flipped in serving).

Usage: drive_replica.py <port> <n_requests> [waves]
"""
import json
import sys
import time
import urllib.request

WORDS = ("the quick brown fox jumps over the lazy dog while clouds gather "
         "over silent valleys and rivers carry stones toward the sea under "
         "a sky that remembers every storm it has ever held").split()
SUFFIX = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
          "kilo lima mike november oscar papa quebec romeo sierra tango "
          "uniform victor whiskey xray yankee zulu zero one two three four "
          "five six seven eight nine ten eleven twelve thirteen").split()


def make_prompt(i: int, prefix_words: int) -> str:
    words = [WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(prefix_words)]
    suffix = " ".join(SUFFIX[(i * 5 + j) % len(SUFFIX)]
                      for j in range(24))
    return " ".join(words) + " " + suffix + "\nQuestion: reply with OK."


def call(port: int, prompt: str, max_tokens: int = 1):
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    u = d.get("usage", {})
    return {"sec": round(time.time() - t0, 2),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion": (d["choices"][0]["message"]["content"] or "")[:24]}


def main():
    port = int(sys.argv[1])
    n = int(sys.argv[2])
    waves = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    prefix_words = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    for w in range(waves):
        for i in range(n):
            r = call(port, make_prompt(i, prefix_words))
            print(f"wave{w} req{i:02d}: {r}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
