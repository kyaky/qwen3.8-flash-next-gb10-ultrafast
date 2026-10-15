#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-task floor bench (P0 Stage F): four fixed groups x n>=5 prompts,
C1 and C4 runs, mechanical verification, acceptance telemetry.

Reuses bench_agent.py's stream_chat / spec_delta / summarize / fetch_metrics
verbatim (donor file unmodified; imported, not copied).

Groups:
  code-edit    5 repo-shaped edit tasks, each verified by executing its test
  prose-zhen   5 Chinese + 5 English prose prompts
  copy-agentic 5 tool-loop tasks: lookup_reference call -> tool result (module)
               -> byte-exact copy, verified against the source
  ctx128k      5 tasks at ~131k prompt tokens (needle quote / summary)

Sampling matches the reference eval conditions: temperature 1.0, top_p 0.95, top_k 20,
reasoning_effort medium, max_tokens 1600-2000 (fixed manifest + recorded).
Gate: blended median of the four C1 group medians >= 48.4; no group < 43.8.
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_agent import fetch_metrics, stream_chat, summarize  # noqa: E402

BASE = "http://localhost:8000"

# --------------------------------------------------------------------------
# fixed manifest (recorded; do not regenerate per run)
# --------------------------------------------------------------------------
CODE_EDIT = [
    ("moving_average", "Add a function moving_average(values, window) that returns the list of "
        "windowed averages (floats), truncating the tail. Only output the complete module code."),
    ("flatten_paths", "Add a function flatten(d) that flattens a nested dict into "
        "{'a.b.c': value} dot-paths. Only output the complete module code."),
    ("fix_range_sum", "The sum_upto(n) function is off by one: it should include n itself. "
        "Fix it. Only output the complete module code."),
    ("roman_numerals", "Add a function to_roman(n) for 1..3999 using subtractive notation. "
        "Only output the complete module code."),
    ("top_words", "Add a function top_words(text, k) returning the k most frequent "
        "lowercase words, ties broken alphabetically. Only output the complete module code."),
]

CODE_TEMPLATES = {
    "moving_average": ("def moving_average(values, window):\n    pass\n",
        "assert moving_average([1,2,3,4,5], 2) == [1.5, 2.5, 3.5, 4.5]\n"
        "assert moving_average([5], 1) == [5.0]\n"),
    "flatten_paths": ("def flatten(d):\n    pass\n",
        "assert flatten({'a': {'b': 1}, 'c': 2}) == {'a.b': 1, 'c': 2}\n"
        "assert flatten({'x': {'y': {'z': 3}}}) == {'x.y.z': 3}\n"),
    "fix_range_sum": ("def sum_upto(n):\n    return sum(range(1, n))\n",
        "assert sum_upto(5) == 15\nassert sum_upto(1) == 1\n"),
    "roman_numerals": ("def to_roman(n):\n    pass\n",
        "assert to_roman(9) == 'IX'\nassert to_roman(1994) == 'MCMXCIV'\n"
        "assert to_roman(3999) == 'MMMCMXCIX'\n"),
    "top_words": ("def top_words(text, k):\n    pass\n",
        "assert top_words('a b a c a b', 2) == ['a', 'b']\n"
        "assert top_words('b a', 2) == ['a', 'b']\n"),
}

PROSE = [
    ("zh-1", "用中文写一段约300字的短文,主题:城市夜间经济的利与弊。要求观点平衡,结构清晰。"),
    ("zh-2", "用中文解释什么是复式记账法,面向没有会计背景的读者,约300字。"),
    ("zh-3", "用中文写一封简短的商务邮件,向客户解释项目延期两周的原因和补救措施。"),
    ("zh-4", "用中文概述太极拳的历史与健身价值,约250字。"),
    ("zh-5", "用中文写一段产品说明,介绍一款便携咖啡机的核心卖点,约200字。"),
    ("en-1", "Write a ~300-word essay on the trade-offs of remote work for software teams."),
    ("en-2", "Explain how a heat pump works to a non-technical reader, ~300 words."),
    ("en-3", "Write a short release note announcing version 2.0 of a CLI tool, friendly but precise."),
    ("en-4", "Summarize the causes and consequences of the 1973 oil embargo in ~250 words."),
    ("en-5", "Write a ~200-word product description for a mechanical keyboard aimed at writers."),
]

COPY_SYSTEM = ("You are a coding assistant.\n" + "\n".join(
    f"Reference instruction {i:03d}: Preserve source code exactly when asked to copy it. "
    "Keep indentation, identifiers, literals, comments and line order unchanged. "
    "Do not add explanations, markdown fences or tool calls." for i in range(1, 116)))
COPY_TOOLS = [{"type": "function", "function": {
    "name": "lookup_reference", "description": "Look up a public programming-language reference entry.",
    "parameters": {"type": "object", "properties": {"topic": {"type": "string"}},
                   "required": ["topic"]}}}]

def _module(variant: int) -> str:
    return "\n".join(
        f"def transform_{variant}_{line:03d}(value): return (value + {line}) * {variant}"
        for line in range(1, 91))

COPY_TASKS = [(f"copy-agentic-{v}", v) for v in (1, 2, 3)]

CTX_FILLER = ("Quarterly audit note {i:05d}: reconciled ledger entries for the regional "
              "distribution center, including freight adjustments, inventory shrinkage "
              "reviews, and carrier invoice disputes. ")

def _ctx(target: int, needle: str, depth: float) -> str:
    reps = max(1, int(target / 42))
    cut = max(0, min(reps - 1, int(reps * depth)))
    parts = []
    for i in range(reps):
        if i == cut:
            parts.append(f"\n{needle}\n ")
        parts.append(CTX_FILLER.format(i=i))
    return "".join(parts)

CTX_TASKS = [
    ("ctx128k-needle-mid", _ctx(131000, "NOTE B1: the vault override code is AX-7Q4F-2026", 0.5),
     "What is the exact vault override code? Reply with ONLY the code.", "AX-7Q4F-2026"),
    ("ctx128k-needle-end", _ctx(131000, "NOTE B2: the escalation contact is Dana R.", 0.9),
     "Who is the escalation contact? Reply with ONLY the name.", "Dana R."),
    ("ctx128k-needle-start", _ctx(131000, "NOTE B3: the fleet PIN is 4417", 0.1),
     "What is the fleet PIN? Reply with ONLY the number.", "4417"),
    ("ctx128k-summary", _ctx(131000, "NOTE B4: audits passed with zero exceptions", 0.5),
     "In one short sentence: what overall outcome do these audit notes describe?", "zero exceptions"),
    ("ctx128k-needle-mid2", _ctx(131000, "NOTE B5: the freight broker is Meridian Logistics", 0.65),
     "Which company is the freight broker? Reply with ONLY the company name.", "Meridian Logistics"),
]

GROUPS = {
    "code-edit": [t[0] for t in CODE_EDIT],
    "prose-zhen": [t[0] for t in PROSE],
    "copy-agentic": [t[0] for t in COPY_TASKS],
    "ctx128k": [t[0] for t in CTX_TASKS],
}


class Cfg:
    def __init__(self, base, effort, max_tokens, system, tools):
        self.base = base; self.model = "qwen"; self.system = system
        self.max_tokens = max_tokens; self.effort = effort
        self.tools = tools; self.request_timeout = 1200


def stream_chat_capture(cfg, task_id: str, task_user: str) -> dict:
    """stream_chat (same fields/metrics) plus the full completion text, for
    mechanical verification of code-edit tasks."""
    import urllib.request
    from bench_agent import build_body
    sample = {"task": task_id, "ok": False}
    before = fetch_metrics(cfg.base)
    text_parts, reason_parts = [], []
    ttft = None; usage = None; chunks = 0
    req = urllib.request.Request(
        cfg.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(build_body(cfg, task_user)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=cfg.request_timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                chunks += 1
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    c = delta.get("content") or ""
                    if (r or c) and ttft is None:
                        ttft = time.perf_counter() - t0
                    reason_parts.append(r); text_parts.append(c)
        sample["ok"] = True
    except Exception as exc:
        sample["error"] = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - t0
    after = fetch_metrics(cfg.base)
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    completion_tokens = usage.get("completion_tokens")
    sample.update({"ttft_s": ttft, "total_s": total, "chunks": chunks,
                   "reasoning_chars": sum(map(len, reason_parts)),
                   "content_chars": sum(map(len, text_parts)),
                   "prompt_tokens": usage.get("prompt_tokens"),
                   "completion_tokens": completion_tokens,
                   "cached_tokens": details.get("cached_tokens"),
                   "decode_tok_s": None,
                   "_text": "".join(text_parts)})
    if completion_tokens and ttft is not None and total > ttft:
        sample["decode_tok_s"] = completion_tokens / (total - ttft)
    sample["spec"] = spec_delta_import()(before, after)
    return sample


def spec_delta_import():
    from bench_agent import spec_delta
    return spec_delta


def verify_code_edit(name: str, text: str) -> bool:
    """Execute the returned module with the task's asserts in a clean interpreter."""
    template, asserts = CODE_TEMPLATES[name]
    code = text.strip()
    if code.startswith("```"):
        code = "\n".join(code.splitlines()[1:-1]) if code.endswith("```") else code.splitlines()[1]
    program = template + "\n" + code + "\n" + asserts + "\nprint('VERIFY_OK')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(program); path = fh.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=60)
        return "VERIFY_OK" in r.stdout
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def run_copy_agentic(task_id: str, variant: int) -> dict:
    """Tool-loop: request -> expect lookup_reference call -> inject module as tool
    result -> expect byte-exact copy. Returns a merged sample over both turns."""
    cfg = Cfg(BASE, "medium", 2000, COPY_SYSTEM, COPY_TOOLS)
    user = (f"Use the lookup_reference tool with topic 'module_{variant}' to fetch the "
            f"reference module, then copy it exactly as plain text.")
    t0 = time.perf_counter()
    body = {"model": "qwen",
            "messages": [{"role": "system", "content": COPY_SYSTEM},
                         {"role": "user", "content": user}],
            "tools": COPY_TOOLS, "tool_choice": "auto",
            "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": 400, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"reasoning_effort": "medium"}}
    import urllib.request
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    before = fetch_metrics(BASE)
    tool_call = None
    usage1 = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        buf = []
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
            except ValueError:
                continue
            if obj.get("usage"):
                usage1 = obj["usage"]
            for ch in obj.get("choices") or []:
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    while len(buf) <= idx:
                        buf.append({"name": "", "args": ""})
                    if tc.get("function", {}).get("name"):
                        buf[idx]["name"] += tc["function"]["name"]
                    if tc["function"].get("arguments"):
                        buf[idx]["args"] += tc["function"]["arguments"]
    if not buf or buf[0]["name"] != "lookup_reference":
        return {"task": task_id, "ok": False, "error": f"no lookup_reference call: {buf[:1]}"}
    tool_call = buf[0]
    module = _module(variant)
    mid = time.perf_counter() - t0
    # second turn with the tool result
    body["messages"] += [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup_reference", "arguments": tool_call["args"] or "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": module},
    ]
    body["max_tokens"] = 2600
    req2 = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    content = []
    usage2 = {}
    with urllib.request.urlopen(req2, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
            except ValueError:
                continue
            if obj.get("usage"):
                usage2 = obj["usage"]
            for ch in obj.get("choices") or []:
                c = (ch.get("delta") or {}).get("content") or ""
                if c:
                    content.append(c)
    total = time.perf_counter() - t0
    text = "".join(content).strip()
    exact = module in text
    ptoks = (usage1.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0))
    ctoks = (usage1.get("completion_tokens", 0) + usage2.get("completion_tokens", 0))
    after = fetch_metrics(BASE)
    from bench_agent import spec_delta
    sample = {"task": task_id, "ok": exact, "byte_exact": exact,
              "ttft_s": None, "total_s": round(total, 3), "turn1_s": round(mid, 3),
              "prompt_tokens": ptoks, "completion_tokens": ctoks,
              "cached_tokens": (usage2.get("prompt_tokens_details") or {}).get("cached_tokens"),
              "e2e_tok_s": (ctoks / (total)) if total else None,  # two-turn end-to-end, NOT TTFT-excluded decode
              "spec": spec_delta(before, after)}
    return sample


def run_single(group: str, idx: int, base: str) -> dict:
    """Run exactly one task of a group by index (used for C4 concurrency)."""
    if group == "code-edit":
        name, task = CODE_EDIT[idx % len(CODE_EDIT)]
        cfg = Cfg(base, "medium", 1600, "You are a precise coding assistant. Output only complete, runnable Python module code, no fences, no prose.", None)
        return stream_chat_capture(cfg, name, task)
    if group == "prose-zhen":
        tid, task = PROSE[idx % len(PROSE)]
        cfg = Cfg(base, "medium", 1200, "You are a helpful writing assistant.", None)
        return stream_chat(cfg, tid, task)
    if group == "copy-agentic":
        tid, v = COPY_TASKS[idx % len(COPY_TASKS)]
        return run_copy_agentic(tid, v)
    if group == "ctx128k":
        tid, ctx, q, want = CTX_TASKS[idx % len(CTX_TASKS)]
        cfg = Cfg(base, "low", 256, "You are a precise assistant. Follow the instruction exactly.", None)
        s = stream_chat_capture(cfg, tid, ctx + "\n\n" + q)
        text = s.pop("_text", "")
        s["answer"] = text[:200]
        s["verified"] = bool(text and want.lower() in text.lower())
        s["ok"] = s.get("ok", False) and s["verified"]
        return s
    raise SystemExit(f"unknown group {group}")


def run_group(group: str, streams: int, base: str) -> list:
    n = len(GROUPS[group])
    return [run_single(group, i, base) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="real-tasks.json")
    args = ap.parse_args()

    all_samples = []
    n_tasks = len(GROUPS[args.group])
    for rep in range(args.repeats):
        if args.streams == 1:
            samples = run_group(args.group, 1, BASE)
            # attach verification for code-edit
            if args.group == "code-edit":
                for s in samples:
                    name = s.get("task")
                    text = s.pop("_text", "")
                    s["verify_ok"] = verify_code_edit(name, text) if text else None
            agg_ct = sum(s.get("completion_tokens") or 0 for s in samples)
            rates = [s.get("decode_tok_s") for s in samples if s.get("decode_tok_s")]
            print(f"rep{rep}: C1 med={statistics.median(rates) if rates else None} "
                  f"ok={sum(1 for s in samples if s.get('ok'))}/{len(samples)}")
            all_samples.append({"rep": rep, "samples": samples,
                                "group_median": statistics.median(rates) if rates else None})
        else:
            results = [None] * n_tasks
            before = fetch_metrics(BASE)
            def worker(i):
                results[i] = run_single(args.group, i, BASE)
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_tasks)]
            t0 = time.perf_counter()
            for th in threads: th.start()
            for th in threads: th.join()
            wall = time.perf_counter() - t0
            after = fetch_metrics(BASE)
            from bench_agent import spec_delta
            ct = sum(r.get("completion_tokens") or 0 for r in results if r)
            agg = {"rep": rep, "streams": args.streams, "wall_s": round(wall, 3),
                   "aggregate_tok_s": round(ct / wall, 1) if wall else None,
                   "spec": spec_delta(before, after), "samples": results}
            all_samples.append(agg)
            print(f"rep{rep}: C{args.streams} aggregate={agg['aggregate_tok_s']} tok/s")

    json.dump(all_samples, open(args.out, "w"), indent=1, ensure_ascii=False, default=str)
    rates = [s.get("group_median") for s in all_samples if isinstance(s.get("group_median"), (int, float))]
    print(f"GROUP {args.group}: C1 median-of-medians = "
          f"{statistics.median(rates) if rates else None} tok/s over {len(rates)} reps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
