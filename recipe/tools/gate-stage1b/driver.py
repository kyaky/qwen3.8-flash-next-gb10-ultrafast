#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-1b gate harness for the GB10 adaptive draft-width work (P2).

Implements the five-point gate demanded by the codex design review D-009
(task-mv6hbxkt, section d) as concrete, runnable assertions:

  P1 direct-work evidence on a forced-K boot (launch K=4, forced proposal K=3)
  P2 in-process 4->3->4 transitions with bs>1 and request reordering
  P3 boundary coverage (first proposal, chunked/mixed prefill, shortened
     windows, rejection, finish/reuse) with async scheduling asserted EXCLUDED
  P4 token-ID comparison vs the MTP=3 baseline + stochastic distribution gate
  P5 armed-drive verification (the armed decision reaches the drafter)

Subcommands:
  selftest (default)  validate the harness logic against MOCKED metrics, logs
                      and server responses. No server, no docker, no writes
                      outside the report file. Safe to run at any time.
  check-live          READ-ONLY: one /metrics scrape + container argv inspect.
                      Reports which counters exist (including the new stage-1b
                      debug counter). Sends no workload, boots nothing.
  capture-baseline    record the P4 baseline token-ID outputs from the
                      currently running MTP=3 / mode=off boot. Sends a small
                      fixed prompt set (NOT a benchmark). Requires --live-ok.
  run                 execute gate points. BOOTS REPLACE CONTAINERS.
                      Requires --allow-boot and an installed stage-1b patch.

Design notes:
  - All server/boot interactions go through small source interfaces
    (MetricsSource / LogSource / ServerClient) so the selftest injects mocks
    through the exact same code paths the live runners use. In the mock world
    the SERVER drives the simulation: each chat advances the mock engine N
    steps; /metrics fetches are pure snapshots.
  - All edits the harness makes to launch config dirs are assert-guarded
    (the project has had two silent s.replace misses): every anchored rewrite
    asserts its match count and the harness refuses to boot on mismatch.
  - The new debug counter the stage-1b patch must add is specified in
    README.md (section "Debug-counter contract"); the harness treats its
    absence as MISSING_INSTRUMENT, never as a silent pass. A PRESENT but
    MALFORMED emitter payload (a JSON value of the wrong type where the C2
    contract requires an array/object) is MISSING_MALFORMED_INSTRUMENT and
    is type-checked WITHOUT being iterated, so a buggy emitter can never
    crash the gate run - it fails its check (round-3 N5). The P1 alignment
    check additionally requires the reqs payload itself and cross-checks the
    worker row count against the paired sched bs (round-3 N6).

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "gate_config.json"

# ---------------------------------------------------------------------------
# Fixed contract names (see README.md "Debug-counter contract"). Do not rename
# here without renaming there: the stage-1b implementer adds exactly these.
# ---------------------------------------------------------------------------
TAG = "AWDBG1B"                        # evidence-line tag (logger.info)
ENV_EVIDENCE = "VLLM_SPECDBG_EVIDENCE"  # env gate for both emitters (default 0)
# Round 8 (P2 stage-1 closure, D-014 condition 1): the runtime BEHAVIORAL
# fingerprint tag (print, one JSON object with a "kind" field per line:
# CODE/CONFIG/PROBE/POLICY/DRIFT). Emitted by awfp_fingerprint.py on every
# boot of an iter7f-or-later image; a boot without these lines is
# MISSING_INSTRUMENT to the FP checks -- the fingerprint's absence is
# itself loud.
TAG_AWFP = "AWFP"
ENV_AWFP = "VLLM_AWFP"

# Existing per-engine spec-decode counters (prometheus exposition appends
# "_total" to Counter names).
C_DRAFTS = "vllm:spec_decode_num_drafts_total"
C_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
C_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
C_ACCEPTED_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
C_PREFIX_HITS = "vllm:prefix_cache_hits_total"
# NEW (stage-1b): direct-work counter, one increment per actually-executed
# drafter sample per (request, draft position). Position labels 0..launch_k-1.
C_SAMPLES_POS = "vllm:spec_decode_num_sampled_draft_tokens_per_pos_total"

INSTALL_SNIPPET = "adaptive width installed: mode="
ARMED_SNIPPET = "adaptive width armed: driving per-step K"
# Round 5: the patched scheduler's update_draft_token_ids logs this ONCE on
# its first call. That method only runs when the engine core pulls the
# post-step DraftTokenIds channel -- i.e. when async scheduling is OFF. The
# round-4 gate run reported "async flags: none" from an argv-only grep while
# SchedulerConfig.async_scheduling had AUTO-resolved to True for the MTP boot
# (async_scheduling=None + "mtp" in EagleModelTypes + an async-capable
# executor), the AsyncScheduler ran, this channel never executed, and the C1
# counter stayed at a zero delta. P3.S7 now asserts THIS line (engine truth),
# with the argv grep demoted to a secondary signal.
SYNC_CHANNEL_SNIPPET = "sync draft-token channel live"
# The flag the harness injects into the copied config's EXTRA so the gate
# boots really run the documented sync path (the D-009 contract excludes
# async scheduling pending verification).
NO_ASYNC_FLAG = "--no-async-scheduling"
# Prefix-cache block size the P3.S6 probe must exceed (one hash block; the
# P0 agentic20-v11 probe recorded 1,600-token hits).
PREFIX_BLOCK_TOKENS = 1600

# Round 6: the round-5 review landed S7's unfiltered argv substring grep
# ("async" in token) in the same round that made every harness boot append
# --no-async-scheduling to the env EXTRA -- so on every harness boot the
# grep matched the harness's OWN negated flag, S7 self-tripped, and P3
# (hence OVERALL) false-FAILed with the self-contradictory detail naming
# the exclusion flag as an async flag. The signal S7 wants is an ENABLE
# flag: --async-scheduling (argparse BooleanOptionalAction derives
# --no-async-scheduling as its negation; both forms are valid on the
# iter6d and iter7a vLLM). Negated forms are excluded; any OTHER
# async-bearing token still counts (fail-noisy: an unknown async knob must
# fail loudly, not silently).
_ASYNC_NEG_RE = re.compile(r"^--no[-_]async[-_]scheduling(?:=.*)?$", re.IGNORECASE)
_ASYNC_POS_RE = re.compile(r"^--async[-_]scheduling(?:=.*)?$", re.IGNORECASE)


def argv_async_flags(argv) -> list[str]:
    """Async-scheduling ENABLE flags in a container argv (negations excluded)."""
    flags = []
    for a in argv:
        tok = str(a).strip()
        if _ASYNC_NEG_RE.match(tok):
            continue
        if _ASYNC_POS_RE.match(tok) or "async" in tok.lower():
            flags.append(tok)
    return flags

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
MISSING_INSTRUMENT = "MISSING_INSTRUMENT"  # a FAIL caused by absent instrumentation
# A FAIL caused by instrumentation that is PRESENT but malformed: a JSON value
# of the wrong type where the C2 contract requires an array/object (round-3
# N5). Same FAIL-class aggregation as MISSING_INSTRUMENT (a malformed emitter
# payload must fail the CHECK, never crash the gate run); the distinct label
# says the emitter exists but is emitting the wrong shape.
MISSING_MALFORMED_INSTRUMENT = "MISSING_MALFORMED_INSTRUMENT"

# Prometheus sample line grammar (same as recipe/benchmarks/bench_agent.py;
# kept local so the harness is standalone).
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)\s*$")
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------------------
# metrics parsing / delta helpers
# ---------------------------------------------------------------------------
def parse_metrics(text: str) -> dict:
    """Parse Prometheus text exposition into {name: [(labels, float), ...]}."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue  # NaN / +Inf / malformed -- not usable as a counter
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        out.setdefault(m.group("name"), []).append((labels, value))
    return out


def msum(metrics: dict, name: str, **want) -> float:
    total = 0.0
    for labels, value in (metrics or {}).get(name, []):
        if all(labels.get(k) == v for k, v in want.items()):
            total += value
    return total


def delta(before: dict, after: dict, name: str, **want) -> float:
    return msum(after, name, **want) - msum(before, name, **want)


def pos_series(metrics: dict, name: str, positions, engine: str = "0") -> dict:
    return {p: msum(metrics, name, engine=engine, position=str(p)) for p in positions}


def pos_delta(before: dict, after: dict, name: str, positions, engine: str = "0") -> dict:
    b = pos_series(before, name, positions, engine)
    a = pos_series(after, name, positions, engine)
    return {p: a[p] - b[p] for p in positions}


# ---------------------------------------------------------------------------
# sources (live + mock)
# ---------------------------------------------------------------------------
class MetricsSource:
    def fetch(self) -> dict:
        raise NotImplementedError

    def now(self) -> float:
        return time.time()

    def sleep(self, s: float) -> None:
        time.sleep(s)


class LiveMetrics(MetricsSource):
    def __init__(self, base: str, timeout: float = 15.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def fetch(self) -> dict:
        with urllib.request.urlopen(self.base + "/metrics", timeout=self.timeout) as r:
            return parse_metrics(r.read().decode("utf-8", "replace"))


class LogSource:
    def since(self, unix_ts: float) -> list[tuple[float, str]]:
        raise NotImplementedError


class DockerLogs(LogSource):
    def __init__(self, container: str):
        self.container = container

    def since(self, unix_ts: float) -> list[tuple[float, str]]:
        out = subprocess.run(
            ["docker", "logs", "--timestamps", "--since", f"{max(0, unix_ts - 1):.0f}",
             self.container],
            capture_output=True, text=True, timeout=120)
        lines = []
        for ln in (out.stdout + out.stderr).splitlines():
            m = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+Z) ?(.*)$", ln)
            if not m:
                continue
            lines.append((_rfc3339_to_unix(m.group(1)), m.group(2)))
        return lines


def _rfc3339_to_unix(s: str) -> float:
    import calendar

    t = calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))
    frac = 0.0
    if "." in s:
        frac = float("0." + s.split(".", 1)[1].rstrip("Z"))
    return t + frac


class MockLogs(LogSource):
    def __init__(self, lines: list[tuple[float, str]]):
        self.lines = lines

    def since(self, unix_ts: float) -> list[tuple[float, str]]:
        return [(t, l) for (t, l) in self.lines if t >= unix_ts]


class ServerClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(self, base: str, model: str, timeout: float = 600.0):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, max_tokens=64, temperature=0.0, seed=None,
             stream=False, extra=None) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if seed is not None:
            body["seed"] = seed
        if extra:
            body.update(extra)
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + "/v1/chat/completions", data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if stream:
                return self._read_stream(r)
            return json.loads(r.read())

    def chat_token_ids(self, prompt: str, max_tokens=64, temperature=0.0,
                       seed=None) -> list[int]:
        """Completion token-ID sequence via streaming return_token_ids."""
        return self.chat_token_ids_tracked(prompt, max_tokens=max_tokens,
                                           temperature=temperature,
                                           seed=seed)[0]

    def chat_token_ids_tracked(self, prompt: str, max_tokens=64,
                               temperature=0.0, seed=None) -> tuple[list[int], str | None]:
        """(token IDs, response id) via streaming return_token_ids.

        The response id is the exact C2 chain-attribution key: the evidence
        req ids are `<response id>-<suffix>`, so a capture window can pick
        ITS OWN request's committed-prefix chain instead of the longest
        chain that happens to fall in the window (the round-8 first-
        appearance protocol requires an ol=0 record, which only chunked
        prefills emit -- a single-chunk request's first commit record
        starts at ol>=1, and a previous request's tail lines can bleed
        into the next window through the log-write race).
        """
        resp = self.chat([{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature,
                         seed=seed, stream=True, extra={"return_token_ids": True})
        ids: list[int] = []
        resp_id: str | None = None
        for chunk in resp["chunks"]:
            if resp_id is None:
                resp_id = chunk.get("id")
            for choice in chunk.get("choices", []):
                tok = choice.get("token_ids")
                if tok:
                    ids.extend(int(x) for x in tok)
        return ids, resp_id

    @staticmethod
    def _read_stream(resp) -> dict:
        chunks = []
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunks.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
        return {"chunks": chunks}


class MockServer(ServerClient):
    """Scripted streaming responses; each chat advances the mock world."""

    def __init__(self, world, outputs: dict[str, list[int]] | None = None,
                 steps_per_call: int = 6, long_prompt_chars: int = 5000):
        super().__init__("http://mock", "qwen")
        self.world = world
        self.outputs = outputs or {}
        self.steps_per_call = steps_per_call
        self.long_prompt_chars = long_prompt_chars
        self.calls: list[dict] = []
        self.seen: set[str] = set()

    def chat(self, messages, max_tokens=64, temperature=0.0, seed=None,
             stream=False, extra=None):
        prompt = messages[-1]["content"]
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens,
                           "temperature": temperature, "seed": seed, "extra": extra or {}})
        # A prompt above the chunking threshold arrives as a chunked prefill:
        # the mock world models it as a joining request with a prefill gap.
        self.world.on_request(steps=self.steps_per_call,
                              long_prefill=len(prompt) > self.long_prompt_chars,
                              prompt=prompt)
        # Outputs are keyed by prompt for greedy (temperature 0) and by
        # prompt#t<temp>#s<seed> for seeded sampling (the same prompt samples
        # differently from its greedy decode).
        if temperature:
            key = f"{prompt}#t{temperature}#s{seed}"
        else:
            key = prompt
        ids = list(self.outputs.get(key,
                                     self.outputs.get(prompt,
                                                      [11] * min(max_tokens, 16))))
        # Round-5 usage model for the S6 probe: tokenized length from the
        # world's chars-per-token model, and a cached-tokens detail on a
        # repeated prompt that clears one prefix-cache block (the world's
        # prefix-hits counter moves with it).
        ptoks = max(10, len(prompt) // self.world.tok_chars_per_token)
        cached = 0
        if prompt in self.seen and ptoks >= self.world.prefix_block:
            cached = ptoks
            self.world._inc(C_PREFIX_HITS, max(1, ptoks // self.world.prefix_block))
        self.seen.add(prompt)
        usage = {"prompt_tokens": ptoks, "completion_tokens": len(ids),
                 "prompt_tokens_details": {"cached_tokens": cached}}
        if stream:
            chunks = []
            step = 4
            for i in range(0, len(ids), step):
                chunks.append({"choices": [{"token_ids": ids[i:i + step]}]})
            chunks.append({"choices": [{"token_ids": []}]})
            return {"chunks": chunks}
        return {"choices": [{"message": {"content": "mock"}, "finish_reason": "stop"}],
                "usage": usage}


# ---------------------------------------------------------------------------
# mock world: behavioral model of one gate boot
# ---------------------------------------------------------------------------
def _mock_prompt_tokens(prompt: str) -> list[int]:
    """Deterministic prompt token ids for the mock's committed-prefix hash
    (the real engine hashes the tokenizer's ids; the mock derives stable
    ids from the prompt text so chains are reproducible across boots)."""
    return [(i * 31 + ord(c) * 7 + 11) % 149983
            for i, c in enumerate(prompt[:64])] or [1]


def _mock_policy_sha(launch_k: int) -> str:
    """The mock's policy hash: stable per LAUNCH SHAPE, so same-shape
    matched boots agree and a differently-shaped boot drifts (the real
    engine hashes its actual cache/kernel selections)."""
    import hashlib as _h
    return _h.sha256(f"policy:{launch_k}".encode()).hexdigest()


def _phash_feed(book: dict, rid: str, prompt_tokens: list[int],
                emitted: list[int]) -> str:
    """The mock's committed-prefix hash (the engine-side commit_hash_update
    mirror: prompt fed once, then each step's committed tokens, int64-le)."""
    import hashlib as _h
    import struct as _s
    entry = book.get(rid)
    if entry is None:
        entry = [_h.sha256(), 0]
        book[rid] = entry
    h, fed = entry
    if fed == 0:
        feed = prompt_tokens + emitted
    else:
        feed = emitted
    if feed:
        h.update(b"".join(_s.pack("<q", int(t)) for t in feed))
    entry[1] = fed + len(feed) if fed else len(feed)
    return h.hexdigest()


class MockWorld:
    """Simulates engine steps of one gate boot: counters, AWDBG1B evidence
    lines, the clock. Scenario knobs select the defect being modeled.

    Semantics modeled (see README, round-4 phase model):
      - requested_k: the per-step proposal width the controller ships
      - k_eff: the width the drafter actually samples (defect scenarios diverge)
      - win: the per-request verify-window length = the PREVIOUS proposal's
        width (one-step lag: the pending verification keeps its own length).
        There is NO launch-K cold-start seeding on the sync path
        (request.spec_token_ids is only ever set from the DraftTokenIds
        output), so a request's FIRST verify window is ALSO the previous
        proposal's width -- win == launch_k on a non-first verify would be
        the pre-stage-1b defect (the launch-width carrier), not an
        invariant. Budget pressure may trim a window below the pending
        width (legitimate; the lag check exempts trims).
      - sampled_per_pos / valid_len / steps: WORKER TRUTH (round 4) -- the
        ar drafter executes one sample per batch row at EVERY position
        0..k_eff-1 (first-pass sample + each _generate_draft call), so
        valid_len == k_eff and steps == range(k_eff) on EVERY row,
        including rows without a scheduled verify window (chunked prefill
        consumes its samples; only its scheduler-side window stays empty),
        and sampled_per_pos is the speculator's executed-sample counter
        ([bs]*k_eff + zeros on this drafter).
    """

    STEP_DT = 0.5  # mock seconds per engine step

    def __init__(self, scenario: str, launch_k: int = 4, force_k: int = 3,
                 bs: int = 4, t0: float | None = None):
        self.scenario = scenario
        self.launch_k = int(launch_k)
        self.force_k = int(force_k)
        self.bs = int(bs)
        self._t = (time.time() - 120.0) if t0 is None else t0
        self.t0 = self._t
        self.step_no = 0
        self._counters: dict = {}
        self.log_lines: list[tuple[float, str]] = []
        self.req_counter = 0
        # active: [{id, prefill_left, pending}]; pending = the width of the
        # request's last KEPT proposal (None until its prefill chunks are
        # done -- a chunked-prefill row's proposals are discarded
        # scheduler-side, update_draft_token_ids ignores prefill chunks).
        self.active: list[dict] = []
        self.accept_profile = (0.7, 0.45, 0.2)  # per-pos acceptance fractions
        self.new_counter = True
        self.evidence_on = True
        self.install_ok = True
        self.armed_ok = True
        self.window_stall = scenario == "window-stall"
        self.unpair_worker = scenario == "unpaired-worker"
        # Round 5 scenarios:
        # - warmup-proposes: the runner-side JIT kernel warmup emitted six
        #   boot-time worker lines the scheduler never paired (the live
        #   2026-10-14 defect: worker seq ran 6 ahead of sched seq for the
        #   whole boot). The FIXED engine gates the emitter to scheduled
        #   batches, so healthy scenarios emit none.
        # - c1-silent: the C1 counter is present but never increments (the
        #   async-dead staging transport) while the evidence lines carry
        #   samples -- A3/A3c must fail on the zero deltas.
        # - async-auto: argv is CLEAN (no --async-scheduling flag) but the
        #   AsyncScheduler is live (the auto-enable), so the sync
        #   draft-token channel never logs -- P3.S7's engine-truth probe
        #   must catch what the argv grep cannot.
        # - s6-short: the S6 probe's tokenized length collapses below one
        #   prefix-cache block -- a probe-shape defect S6's length gate must
        #   report instead of passing vacuously.
        self.worker_seq_offset = 6 if scenario == "warmup-proposes" else 0
        self.c1_dead = scenario == "c1-silent"
        self.p3_like = scenario in ("p3", "s6-short", "s4-flat")
        # Round 7: the P3.S4 trim model is SHAPE-DRIVEN. The retuned S4 shape
        # calibrates its long prompt so the prefill chunk leaves only a few
        # tokens of budget; while a long-prefill row is chunking, the last
        # decode row behind it takes a partial window (the trim the check
        # detects). s4-flat models the UNTUNED shape (the chunk leaves a huge
        # leftover): no trims, S4 reports the coverage gap as WARN.
        self.s4_trim = scenario == "p3"
        # Round 6: token-budget ROW trims. row-trim / osc-row-trim model the
        # engine-legitimate shape the round-6 B4/A8/A3c exemption accepts:
        # every 5th step the last row is trimmed to 0 tokens by the token
        # budget -- absent from num_scheduled_tokens (so absent from the
        # sched line's reqs and its bs, win == 0 by absence) while its row
        # persists in the worker batch and samples like every row. row-drop /
        # osc-row-drop model the DEFECT the exemption must NOT mask: the
        # worker emitter drops a SCHEDULED row from its line (the sched line
        # keeps counting it), which row accounting must still fail.
        self.row_trim = scenario in ("row-trim", "osc-row-trim")
        self.row_drop = scenario in ("row-drop", "osc-row-drop")
        # Round 8 (D-014 condition 1): the AWFP fingerprint layer. Healthy
        # boots emit CODE/CONFIG/PROBE at init and POLICY at scheduled steps
        # 1 and 32. Scenario knobs:
        # - no-awfp: the fingerprint layer is ABSENT (a pre-iter7f image) --
        #   the FP checks must report MISSING_INSTRUMENT, never pass.
        # - awfp-probe-fail: the behavioral shim probe FAILED (the shim is
        #   absent, unwired or env-gated off) -- the FP probe check fails.
        # - awfp-code-bad: the marlin code-state check failed (the
        #   atomic_add call-site proof or the shim wiring is broken).
        self.awfp_on = scenario not in ("no-awfp", "no-counter")
        self.awfp_probe_ok = scenario != "awfp-probe-fail"
        self.awfp_code_ok = scenario != "awfp-code-bad"
        # The mock's committed-prefix hash book (the engine-side mirror:
        # one sha256 hasher + the fed length per request, prompt fed once,
        # then each step's committed tokens).
        self.phash_book: dict = {}
        # tokenized-length model for MockServer usage reports (chars per
        # token); s6-short models a probe that tokenizes tiny.
        self.tok_chars_per_token = 10**6 if scenario == "s6-short" else 4
        self.prefix_block = PREFIX_BLOCK_TOKENS
        self.sync_channel = scenario not in ("async-enabled", "async-auto",
                                             "no-counter")
        if self.awfp_on:
            self._emit_awfp_boot_lines()
        if scenario == "no-counter":
            # stage-1b patch absent: no new counter, no evidence emitters
            self.new_counter = False
            self.evidence_on = False
        if scenario == "no-evidence":
            # round 7: the emitters are installed but SILENT (the env gate
            # off, or an emitter that never fires): the counter exists, the
            # AWDBG1B lines do not. A0 must report MISSING_INSTRUMENT even
            # though the counter half looks healthy - and because the
            # boot-manager instrument probe guarantees a request ran first,
            # that report is a real emitter fault, not the fresh-boot
            # artifact the 2026-10-14 round-7 gate run produced.
            self.evidence_on = False
        if scenario == "no-install":
            # controller install failed (e.g. candidates above launch_k):
            # silently falls back to static K
            self.install_ok = False
        if scenario == "dist-drift":
            # stale/shifted proposal distributions: the acceptance profile
            # moves (the D-009 q-vs-stale-q' defect class greedy cannot see)
            self.accept_profile = (0.9, 0.25, 0.15)
        self._emit_install_lines()
        if self.new_counter:
            # A real Prometheus counter is pre-created at init: it appears in
            # /metrics at ZERO before the first increment (the c1-silent
            # scenario models exactly that state -- present but dead).
            for p in range(self.launch_k):
                self._inc(C_SAMPLES_POS, 0.0, {"position": str(p)})
        if self.worker_seq_offset:
            self._emit_warmup_worker_lines()
        if self.sync_channel:
            # the patched sync draft-token channel logs once, on its first
            # update_draft_token_ids call (the boot's first scheduled step --
            # after the runner-side warmup, before any traffic step)
            self.log_lines.append((self._t - 0.3, (
                "INFO 10-13 00:00:00 [scheduler.py:1] GB10 stage-1b: "
                + SYNC_CHANNEL_SNIPPET
                + " (update_draft_token_ids runs; async scheduling off)")))
        for i in range(self.bs):
            prefill_left = 3 if (scenario == "osc-prefill" and i == 0) else 1
            self.active.append(self._new_req("boot-probe"))
        # A gate boot is never cold at the counters: the boot manager's
        # instrument probe (round 7: one tiny request right after readiness,
        # because the request-driven emitters emit nothing on a fresh boot)
        # has already produced spec-decode steps before any point runs. The
        # mock models those steps with this advance.
        self.advance(12)

    def _emit_warmup_worker_lines(self):
        """Round-5 defect model: the six JIT-kernel-warmup worker lines the
        round-4 emitter produced at boot (rows 8/8/2/2/1/1, k_eff == the
        launch width, requested 0 -> no sched line ever). They consume worker
        seq values 1..6 so every later worker line is offset against the
        sched seq counter -- exactly the live gate-run signature. Timestamps
        sit after the install lines (the controller installs at scheduler
        init, before the worker warmup) and before the first step."""
        rows = [self.bs, self.bs, 2, 2, 1, 1]
        t = self._t - 1.5
        for i, nrows in enumerate(rows):
            wrec = {
                "src": "worker", "seq": i + 1, "t": int(t * 1e6),
                "k_eff": self.launch_k,
                "sampled_per_pos": [nrows if p < self.launch_k else 0
                                    for p in range(self.launch_k)],
                "reqs": [{"id": f"_warmup_{j}_", "valid_len": self.launch_k,
                          "row": j, "steps": list(range(self.launch_k))}
                         for j in range(nrows)],
            }
            self.log_lines.append(
                (t, "INFO [gpu/model_runner] " + TAG + " " + json.dumps(wrec)))
            t += 0.2

    # -- setup
    def _emit_install_lines(self):
        if not self.install_ok:
            return
        cands = sorted({self.force_k, self.launch_k})
        mode = "force" if "osc" not in self.scenario else "armed"
        force = self.force_k if mode == "force" else "oscillate"
        t = self._t - 2.0
        self.log_lines.append((t, (
            f"INFO 10-13 00:00:00 [vllm_adaptive_width.py:1] {INSTALL_SNIPPET}"
            f"mode={mode} force={force} candidates={cands} launch_k={self.launch_k}")))
        if mode == "armed" and self.armed_ok:
            self.log_lines.append((t + 0.1, (
                "INFO 10-13 00:00:00 [scheduler.py:1] " + ARMED_SNIPPET
                + f" (mode=armed force=oscillate k={self.launch_k} "
                  f"launch_k={self.launch_k})")))

    def _new_id(self) -> str:
        self.req_counter += 1
        return f"r{self.req_counter:03d}"

    def _new_req(self, prompt: str) -> dict:
        """One tracked request: the round-8 C2 state the engine keeps
        per request (the committed-prefix hash is fed the prompt once,
        then each step's committed tokens; proposals derive from the
        PROMPT so chains are reproducible across mock boots)."""
        rid = self._new_id()
        return {"id": rid, "pending": None, "prefill_left": 1, "long": False,
                "prompt": _mock_prompt_tokens(prompt), "props": None,
                "ol": 0, "temp": 0.0, "seed": 0, "seed_set": False}

    def _emit_awfp_boot_lines(self) -> None:
        """The boot-time AWFP records the iter7f fingerprint layer emits:
        CODE (marlin import time), CONFIG + PROBE (the load_model tail)."""
        t = self._t - 2.5
        sha = _mock_policy_sha(self.launch_k)
        self.log_lines.append((t, f"{TAG_AWFP} " + json.dumps({
            "kind": "CODE", "t": int(t * 1e6), "ok": self.awfp_code_ok,
            "atomic_add_sites": 2 if self.awfp_code_ok else 1,
            "shim_marker": 1, "shim_call_site": 1,
            "shim_env_on": True, "file": "marlin_moe.py"})))
        self.log_lines.append((t + 0.1, f"{TAG_AWFP} " + json.dumps({
            "kind": "CONFIG", "t": int((t + 0.1) * 1e6),
            "versions": {"torch": "2.13", "triton": "3.x", "vllm": "dev"},
            "env": {"VLLM_MOE_ALIGN_STABLESORT": "1"},
            "spec": {"method": "mtp", "draft_sample_method": "probabilistic"},
            "cudagraph": {"mode": "PIECEWISE",
                          "configured_sizes": [1, 2, 4, 8]}})))
        self.log_lines.append((t + 0.2, f"{TAG_AWFP} " + json.dumps({
            "kind": "PROBE", "t": int((t + 0.2) * 1e6),
            "passed": self.awfp_probe_ok,
            "probe": {"direct": "ok" if self.awfp_probe_ok else "fail",
                      "composed": "ok" if self.awfp_probe_ok else "fail"},
            "code": {"atomic_add_sites": 2, "shim_marker": 1,
                     "shim_call_site": 1, "shim_env_on": True},
            "device": "cuda:0"})))

    # -- clock / snapshot interface used by MockMetrics
    def clock(self) -> float:
        return self._t

    def advance_time(self, dt: float) -> None:
        self._t += dt

    def snapshot(self) -> dict:
        out = {}
        for (name, labels), v in self._counters.items():
            out.setdefault(name, []).append((dict(labels), v))
        return out

    def on_request(self, steps: int, long_prefill: bool = False,
                   prompt: str = "") -> None:
        """One HTTP request arrived: rotate one request (reorder), then advance
        the engine `steps` times. Every request prefills for at least one
        step (the join step); a chunked-prefill request keeps prefilling for
        several steps and its proposals are discarded until the chunks are
        done (no verify windows until then)."""
        self.active.pop(0)
        self.active.append(self._new_req(prompt))
        self.active[-1]["prefill_left"] = 4 if long_prefill else 1
        self.active[-1]["long"] = bool(long_prefill)
        self.active.reverse()  # reorder the batch layout between requests
        self.advance(steps)

    # -- scenario shape
    def _requested(self, step: int) -> int:
        if self.scenario in ("forced3", "trim-defect", "greedy-divergence",
                             "dist-drift", "p3", "s6-short", "s4-flat",
                             "no-evidence", "async-enabled", "async-auto",
                             "row-trim", "row-drop", "align-row-collision",
                             "unpaired-worker", "reqs-empty", "reqs-omitted",
                             "spp-fabricated", "launch-carrier",
                             # round 8: the fingerprint/C2-field defect
                             # scenarios ride the forced rhythm like the
                             # healthy forced boot they defect on
                             "no-awfp", "awfp-probe-fail", "awfp-code-bad",
                             "c2-fields-absent", "c2-props-malformed",
                             "c2-join-missing", "c2-argmax-mismatch",
                             "c2-hash-stale"):
            return self.force_k
        if self.scenario.startswith("osc"):
            cands = sorted({self.force_k, self.launch_k})
            return cands[-1] if step % 2 == 0 else cands[0]
        return self.launch_k

    def _k_eff(self, rk: int) -> int:
        if self.scenario in ("trim-defect", "osc-pin", "launch-carrier"):
            # drafter ignores the per-call width: trim-defect isolates A3b's
            # signature (pos-k samples behind a force_k-wide window);
            # osc-pin / launch-carrier are the stage-1b-consumption-missing
            # state (the launch-width carrier -- windows stay launch_k-wide)
            return self.launch_k
        if self.scenario == "osc-zerowidth":
            # defect: the width derivation collapses the wave's LOW half to
            # a zero width (0 is not a legal width on this stack - B3 must
            # flag the k_eff==0 lines as offenders). The high half still
            # samples, so the debug counter exists and A0 passes.
            cands = sorted({self.force_k, self.launch_k})
            return 0 if rk == cands[0] else rk
        return rk

    # -- engine step
    def advance(self, steps: int) -> None:
        for _ in range(steps):
            self._step_once()

    def _step_once(self) -> None:
        s = self.step_no
        self.step_no += 1
        t = self._t              # evidence stamped with the pre-step clock so
        self._t += self.STEP_DT  # scrape windows [t0, t1] bracket the lines
        rk = self._requested(s)
        k_eff = self._k_eff(rk)
        spp = [0] * self.launch_k
        # Round 8: the proposals the CURRENT step verifies were made at the
        # PREVIOUS step -- snapshot them before the loop overwrites
        # req["props"] with this step's proposal.
        prev_props = {req["id"]: req.get("props") for req in self.active}
        # Round 8: the settled AWFP policy fingerprint at scheduled steps
        # 1 and 32 (the mock mirror of awfp_fingerprint.maybe_emit_policy).
        if self.awfp_on and s in (0, 31):
            sha = _mock_policy_sha(self.launch_k)
            self.log_lines.append((t, f"{TAG_AWFP} " + json.dumps({
                "kind": "POLICY", "t": int(t * 1e6), "policy_sha": sha,
                "step": s + 1, "ffc": {"installed": False},
                "autotune": {"n_autotuners": 3, "kernels": {}},
                "cudagraph": {"mode": "PIECEWISE",
                              "configured_sizes": [1, 2, 4, 8],
                              "captured_sizes": [1, 2, 4, 8]},
                "inductor_entries": 12, "drift": {"drift": None}})))

        # per-request verify windows + per-request draft work. Round-4 phase
        # model: win(t) = the request's pending proposal width (= k_eff(t-1),
        # the one-step lag; trims shrink it), 0 while the request is still
        # prefilling or has no kept proposal yet. There is NO launch-K
        # cold-start seeding on the sync path. WORKER TRUTH: every row
        # samples k_eff positions regardless of its window.
        req_rows = []
        trimmed_idx: set[int] = set()   # unscheduled by the token budget
        dropped_idx: set[int] = set()   # emitter drops the worker line's row
        # Round 6 row-trim scenarios: every 5th step the LAST batch row is
        # trimmed to 0 tokens by the token budget -- absent from
        # num_scheduled_tokens (scheduled_running_reqs only appends actually
        # scheduled requests) while its row persists in the worker batch and
        # samples like every row (worker truth); the sched line then carries
        # bs = rows-1 and the row is engine-legitimate surplus for B4/A8/A3c
        # (the round-6 exemption). row-drop models the DEFECT the exemption
        # must NOT mask: the worker emitter drops a SCHEDULED row from its
        # line (bs keeps counting it), which row accounting must still fail.
        trim_last = self.row_trim and s % 5 == 2 and len(self.active) > 1
        drop_last = self.row_drop and s % 5 == 2 and len(self.active) > 1
        # Round 7: shape-driven S4 trim model (see the s4_trim init comment).
        s4_join = self.s4_trim and any(
            r.get("long") and (r.get("prefill_left") or 0) > 0
            for r in self.active)
        order = list(reversed(self.active)) if s % 3 == 2 else list(self.active)
        for j, req in enumerate(order):
            was_prefill = req.get("prefill_left", 0) > 0
            if was_prefill:
                req["prefill_left"] -= 1
                win = 0                    # prefill chunk: no verify yet
            elif req.get("pending") is None:
                win = 0        # first decode step: its first KEPT proposal
                               # verifies NEXT step (one-step lag)
            elif self.scenario == "trim-defect":
                # D-009 trim-defect signature: the drafter samples the full
                # launch width (k_eff = launch_k) while the scheduler keeps
                # scheduling the width it asked for -- pos-k samples behind
                # a force_k-wide window (A3b's discriminator)
                win = self._requested(s - 1)
            elif self.scenario == "osc-lag-defect":
                win = rk                   # defect: window follows CURRENT k
            elif self.p3_like and s4_join and j == len(order) - 1 \
                    and not was_prefill and req.get("pending"):
                # Round 7: the retuned S4 shape - the calibrated long prompt's
                # prefill chunk leaves only a few tokens of budget behind it,
                # so the LAST decode row (with a pending window) takes a
                # partial window (win < requested_k(t-1), win >= 1). The
                # untuned shape (s4-flat) leaves a huge leftover and never
                # trims - S4 then reports the coverage gap as WARN.
                win = max(1, req["pending"] - 2)  # chunk-boundary budget trim
            elif self.scenario == "osc-trim" and j == 1 and s % 5 == 2:
                win = max(1, req["pending"] - 1)  # legitimate trim
            else:
                win = req["pending"]       # the pending proposal's width
            if j == len(order) - 1 and trim_last and not was_prefill:
                win = 0   # unscheduled this step: no window, no verify events
            # WORKER TRUTH (round 4): every row samples k_eff positions
            # (first-pass sample + each _generate_draft call); rows without
            # a scheduled verify window consume their samples too.
            valid_len = k_eff
            steps = list(range(k_eff))
            if self.scenario == "spp-fabricated":
                # the pre-C1 emitter model: sampled_per_pos / valid_len
                # derived from the scheduler-visible window (rows without
                # one contribute nothing) -- the fabrication the round-4
                # A3c uniform-execution identity exists to catch
                steps = [] if win == 0 else list(range(k_eff))
                valid_len = 0 if win == 0 else k_eff
            if self.scenario == "osc-stale" and k_eff == self.force_k \
                    and s % 2 == 1:
                steps = steps + [self.launch_k - 1]  # stale suffix keeps sampling
            if self.scenario == "osc-column-mix" and k_eff:
                # defect: the per-request draft-step mapping is shifted one
                # column (a permutation, so the AGGREGATE support and the
                # sample-count identity look clean) - only the per-request
                # alignment check can see it
                steps = [((p + 1) % k_eff) for p in steps]
            row = j
            if self.scenario == "align-row-collision" and j == 1:
                row = 0  # defect: two requests share a req_state row
            # Round 8: the row's proposal token ids (deterministic per
            # prompt -- worker truth for the proposal side; prompt-derived
            # so the mock's committed-prefix chains are reproducible across
            # boots, which the D5 identity comparison requires).
            ptoks = req["prompt"]
            props = [(ptoks[p % len(ptoks)] * 7 + p * 3) % 150000
                     for p in range(k_eff)]
            req_rows.append({"id": req["id"], "win": win, "row": row,
                             "valid_len": valid_len, "steps": steps,
                             "req": req, "props": props,
                             "was_prefill": was_prefill})
            if j == len(order) - 1 and trim_last and not was_prefill:
                trimmed_idx.add(j)
            if j == len(order) - 1 and drop_last and not was_prefill:
                dropped_idx.add(j)
            if win:
                self._inc(C_DRAFTS, 1)
                self._inc(C_DRAFT_TOKENS, win)
                prof = self.accept_profile
                acc = 0.0
                for p in range(min(win, self.launch_k)):
                    # positions >= k_eff hold -1 placeholders: never accepted
                    frac = prof[p] if p < min(k_eff, len(prof)) else 0.0
                    self._inc(C_ACCEPTED_POS, frac, {"position": str(p)})
                    acc += frac
                self._inc(C_ACCEPTED, acc)
            # the proposal this row just made is kept (and becomes its next
            # verify window) unless the row was still a prefill chunk --
            # update_draft_token_ids ignores prefill chunks
            if not was_prefill:
                req["pending"] = k_eff
                req["props"] = props  # round 8: the kept proposal's ids

        # worker direct work: spp[p] = sample events at position p across all
        # rows (multiplicity included), derived from the per-request steps so
        # the counters, the evidence lines and the alignment contract stay
        # mutually consistent in every scenario. c1-silent models the round-5
        # live defect: the counter exists but its staging transport never
        # ran (the async scheduler starved update_draft_token_ids), so the
        # deltas stay zero while the evidence lines carry samples.
        if self.new_counter and not self.c1_dead:
            for r in req_rows:
                for p in r["steps"]:
                    if 0 <= p < self.launch_k:
                        spp[p] += 1
            for p, v in enumerate(spp):
                if v:
                    self._inc(C_SAMPLES_POS, v, {"position": str(p)})

        # evidence lines
        if self.evidence_on and not (getattr(self, "window_stall", False) and s >= 3):
            # round 6: the row-drop defect omits a SCHEDULED row from the
            # worker line only (the row sampled; the counters stay complete);
            # the token-budget trim keeps the row in the worker line but the
            # sched line (and its bs) excludes the unscheduled row.
            reqs = [{"id": r["id"], "valid_len": r["valid_len"],
                     "row": r["row"], "steps": r["steps"],
                     # round 8: the proposal-side fields (worker truth)
                     "temp": r["req"]["temp"], "seed": r["req"]["seed"],
                     "seed_set": r["req"]["seed_set"], "props": r["props"]}
                    for i, r in enumerate(req_rows) if i not in dropped_idx]
            wrec = {"src": "worker", "seq": s + 1 + self.worker_seq_offset,
                    "t": int(t * 1e6),
                    "k_eff": k_eff, "sampled_per_pos": spp,
                    # round 8: the executed draft-sampling branch (the mock
                    # models this deployment's probabilistic drafter) and
                    # the execution-shape fingerprint
                    "branch": "gumbel",
                    "nt": sum((r["win"] or 1) for r in req_rows),
                    "pf": any(r["was_prefill"] for r in req_rows),
                    "reqs": reqs}
            if self.unpair_worker and s == 20:
                # stage-1b sign-off: a worker line whose seq pairs with no
                # sched line escapes row accounting (A8b must catch it)
                wrec["seq"] = s + 1 + 9000
            # round-3 defect injections (emission-time only: the counters and
            # the internal steps stay consistent, so the injected defect is
            # the ONLY thing wrong with the line)
            if self.scenario == "reqs-empty":
                wrec["reqs"] = []          # N6: empty reqs payload
            elif self.scenario == "reqs-omitted":
                del wrec["reqs"]           # N6: reqs omitted entirely
            elif self.scenario == "osc-reqs-scalar":
                wrec["reqs"] = 5           # N5: reqs not a JSON array
            elif self.scenario == "osc-steps-scalar":
                # N5: reqs[].steps a scalar instead of an array (the exact
                # round-2 re-review repro: an unhandled TypeError killed the
                # whole gate run at the steps parse)
                wrec["reqs"][0]["steps"] = 5 if s % 2 else 3.0
            elif self.scenario == "c2-fields-absent":
                # round 8: an emitter predating the C2 field contract -- the
                # proposal-side fields are stripped from the worker line
                for k in ("branch", "nt", "pf"):
                    wrec.pop(k, None)
                for rq in wrec["reqs"]:
                    for k in ("temp", "seed", "seed_set", "props"):
                        rq.pop(k, None)
            elif self.scenario == "c2-props-malformed":
                # round 8: a wrongly-typed props value (MISSING_MALFORMED,
                # never a crash)
                wrec["reqs"][0]["props"] = 5
            if self.scenario == "c2-join-missing" and s >= 15 \
                    and s % 7 == 3:
                pass  # the worker line is withheld entirely (below)
            else:
                self.log_lines.append((t, "INFO [gpu/model_runner] " + TAG + " " + json.dumps(wrec)))
            # round 8: the verify/commit-side fields (scheduler truth). For
            # each SCHEDULED row (the token-budget trims are absent from the
            # sched line entirely): the verify outcome (rej_idx = the first
            # rejected position = the accepted count; bonus = the target's
            # own token there) comes from the PREVIOUS step's kept proposal
            # (the one-step lag); the commit side records what entered the
            # request's output list this step and advances the committed-
            # prefix hash over prompt+output.
            sreqs = []
            for i, rr in enumerate(req_rows):
                if i in trimmed_idx:
                    continue
                rid = rr["id"]
                win = rr["win"]
                pprops = prev_props.get(rid)
                if win:
                    rej_idx = max(0, win - 1)
                    if pprops and len(pprops) > rej_idx:
                        bonus = (pprops[rej_idx] + 17) % 150000
                    else:
                        # prompt-derived fallback (chain reproducibility)
                        bonus = (rr["req"]["prompt"][0] * 13 + 7) % 150000
                    emit = list((pprops or [])[:rej_idx]) + [bonus]
                    acc, rej = rej_idx, win - rej_idx
                elif rr["was_prefill"]:
                    rej_idx = bonus = None
                    emit = []      # a prefill chunk commits no output token
                    acc = rej = 0
                else:
                    rej_idx = bonus = None
                    # a plain decode sample (prompt-derived: the mock's
                    # committed-prefix chains must be reproducible across
                    # boots for the D5 identity comparison)
                    emit = [(rr["req"]["prompt"][0] * 11 + 5) % 150000]
                    acc = rej = 0  # a plain decode sample, no verify window
                rr["req"]["ol"] += len(emit)
                ph = _phash_feed(self.phash_book, rid, rr["req"]["prompt"],
                                 emit)
                if self.scenario == "c2-hash-stale" and s % 3 == 1:
                    # round 8 defect: the committed-prefix hash does not
                    # move while the prefix grows (a stale/reused hash) --
                    # the previous step's digest is replayed verbatim
                    ph = rr["req"].get("last_phash") or ph
                rr["req"]["last_phash"] = ph
                sreqs.append({
                    "id": rid, "win": win, "acc": acc, "rej": rej,
                    "rej_idx": rej_idx, "bonus": bonus, "emit": emit,
                    "phash": ph, "ol": rr["req"]["ol"],
                    "nct": len(rr["req"]["prompt"]) + rr["req"]["ol"],
                    # emitter parity: the real CF3 capture records stopped
                    # (and the finishing status name) on EVERY commit row
                    "stopped": False, "stop_reason": None,
                    "temp": rr["req"]["temp"], "seed": rr["req"]["seed"],
                    "seed_set": rr["req"]["seed_set"]})
            if self.scenario == "osc-canonical":
                # defect: a canonical (id-sorted) emitter instead of the
                # pinned batch order - blinds B4b's order-compare
                sreqs = sorted(sreqs, key=lambda r: r["id"])
            if self.scenario == "c2-argmax-mismatch" and s % 3 == 0:
                # round 8 defect: the committed emit diverges from the
                # accepted argmax prefix on a greedy row (the target-logit/
                # argmax evidence fails) -- the emit length and the prefix
                # hash stay consistent, so ONLY the argmax check fires
                for rq in sreqs:
                    if rq["rej_idx"] is not None and rq["emit"]:
                        rq["emit"][0] = (rq["emit"][0] + 1) % 150000
            self.log_lines.append((t + 0.1, "INFO [scheduler] " + TAG + " " + json.dumps(
                {"src": "sched", "seq": s + 1, "t": int((t + 0.1) * 1e6),
                 "launch_k": self.launch_k, "requested_k": rk,
                 "bs": len(sreqs),
                 "mode": "force" if "osc" not in self.scenario else "armed",
                 "reqs": sreqs})))

    def _inc(self, name: str, amount: float, labels: dict | None = None) -> None:
        labels = {"engine": "0", "model_name": "qwen", **(labels or {})}
        key = (name, tuple(sorted(labels.items())))
        self._counters[key] = self._counters.get(key, 0.0) + amount


class MockMetrics(MetricsSource):
    """Pure snapshots; the MockServer drives the world. Clock = world clock."""

    def __init__(self, world: MockWorld):
        self.world = world

    def fetch(self) -> dict:
        return self.world.snapshot()

    def now(self) -> float:
        return self.world.clock()

    def sleep(self, s: float) -> None:
        self.world.advance_time(s)


# ---------------------------------------------------------------------------
# evidence-line parsing (AWDBG1B contract; see README)
# ---------------------------------------------------------------------------
def parse_evidence(lines: list[tuple[float, str]]) -> list[dict]:
    recs = []
    for ts, line in lines:
        idx = line.find(TAG)
        if idx < 0:
            continue
        payload = line[idx + len(TAG):].lstrip(" :\t")
        start = payload.find("{")
        if start < 0:
            continue
        try:
            rec = json.loads(payload[start:])
        except json.JSONDecodeError:
            recs.append({"_bad": True, "ts": ts, "raw": payload[start:start + 200]})
            continue
        rec.setdefault("ts", ts)
        recs.append(rec)
    return recs


def split_src(recs: list[dict], src: str) -> list[dict]:
    return [r for r in recs if r.get("src") == src and not r.get("_bad")]


def join_by_seq(sched: list[dict], worker: list[dict]):
    sm = {r.get("seq"): r for r in sched}
    wm = {r.get("seq"): r for r in worker}
    keys = sorted(set(sm) & set(wm), key=lambda x: (x is None, x or 0))
    pairs = [(sm[k], wm[k]) for k in keys]
    return pairs, [sm[k] for k in sm if k not in wm], [wm[k] for k in wm if k not in sm]


def _reqs_field_problem(w: dict) -> str | None:
    """Type-check a worker line's reqs field WITHOUT iterating it (round-3 N5).

    Returns None when reqs is the C2 contract shape (a JSON array of JSON
    objects); otherwise a short reason: 'absent' (the field is missing or
    null) or 'not a JSON array (<type>)' / 'row N is not a JSON object
    (<type>)'. Consumers that would crash iterating a malformed value use
    this to fail their CHECK with a MISSING_INSTRUMENT / MISSING_MALFORMED_
    INSTRUMENT report instead of killing the whole gate run with a traceback.
    """
    reqs = w.get("reqs")
    if reqs is None:
        return "absent"
    if not isinstance(reqs, list):
        return f"not a JSON array ({type(reqs).__name__})"
    for i, rq in enumerate(reqs):
        if not isinstance(rq, dict):
            return f"row {i} is not a JSON object ({type(rq).__name__})"
    return None


# ---------------------------------------------------------------------------
# round-8 C2 field completion (D-014 condition 2, the astra (b) design):
# proposal-side fields on the worker line, verify/commit-side fields on the
# sched line, the proposal->verify one-step-lag join, the greedy
# target-logit/argmax evidence, the per-request committed-prefix hash chain
# and the semantic cache-state pair. Same discipline as _reqs_field_problem:
# malformed instrumentation fails its CHECK, never the run.
# ---------------------------------------------------------------------------
def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _worker_c2_problem(w: dict) -> str | None:
    """Round-8 worker-field structure check (present + typed), without
    iterating any malformed container.

    Returns None when every round-8 field is present and well-typed
    (branch/nt/pf at top level; temp/seed/seed_set/props on every row);
    otherwise 'absent:<field>' (an emitter predating the field contract)
    or 'malformed:<detail>' (a wrongly-typed value). Callers turn these
    into MISSING_INSTRUMENT / MISSING_MALFORMED_INSTRUMENT reports.
    """
    for key, want in (("branch", str), ("nt", int), ("pf", bool)):
        v = w.get(key)
        if v is None:
            return f"absent:{key}"
        if want is bool:
            if not isinstance(v, bool):
                return f"malformed:{key} is {type(v).__name__}, not a bool"
        elif want is int:
            if not _is_int(v):
                return f"malformed:{key} is {type(v).__name__}, not an int"
        elif not isinstance(v, want):
            return f"malformed:{key} is {type(v).__name__}, not a str"
    if _reqs_field_problem(w) is not None:
        return None  # the reqs shape is the alignment check's finding
    for i, rq in enumerate(w["reqs"]):
        for key in ("temp", "seed", "seed_set", "props"):
            if key not in rq:
                return f"absent:reqs[{i}].{key}"
        if not isinstance(rq["temp"], (int, float)) \
                or isinstance(rq["temp"], bool):
            return f"malformed:reqs[{i}].temp is {type(rq['temp']).__name__}"
        if not _is_int(rq["seed"]):
            return f"malformed:reqs[{i}].seed is {type(rq['seed']).__name__}"
        if not isinstance(rq["seed_set"], bool):
            return (f"malformed:reqs[{i}].seed_set is "
                    f"{type(rq['seed_set']).__name__}, not a bool")
        props = rq["props"]
        if not isinstance(props, list):
            return (f"malformed:reqs[{i}].props is "
                    f"{type(props).__name__}, not a JSON array")
        for t in props:
            if not _is_int(t):
                return (f"malformed:reqs[{i}].props element "
                        f"{type(t).__name__}, not an int")
    return None


def _sched_c2_problem(rq: dict) -> str | None:
    """Round-8 sched-row field structure check for ONE row (typed).

    rej_idx/bonus are null-able (absent verify window); emit/phash/ol/nct
    and the sampling mirror are required on every row. Returns None when
    present and well-typed, else 'absent:<field>' / 'malformed:<detail>'.
    """
    for key in ("rej_idx", "bonus"):
        if key in rq and rq[key] is not None and not _is_int(rq[key]):
            return f"malformed:{key} is {type(rq[key]).__name__}, not an int"
    for key in ("emit", "phash", "ol", "nct"):
        if key not in rq:
            return f"absent:{key}"
    if not isinstance(rq["emit"], list):
        return f"malformed:emit is {type(rq['emit']).__name__}, not a JSON array"
    for t in rq["emit"]:
        if not _is_int(t):
            return f"malformed:emit element {type(t).__name__}, not an int"
    ph = rq["phash"]
    if not isinstance(ph, str) or len(ph) != 64 \
            or any(c not in "0123456789abcdef" for c in ph):
        return "malformed:phash is not a 64-hex sha256 digest"
    for key in ("ol", "nct"):
        if not _is_int(rq[key]) or rq[key] < 0:
            return f"malformed:{key} is {rq[key]!r}, not a non-negative int"
    for key in ("temp", "seed", "seed_set"):
        if key not in rq:
            return f"absent:{key}"
    if not isinstance(rq["temp"], (int, float)) or isinstance(rq["temp"], bool):
        return f"malformed:temp is {type(rq['temp']).__name__}"
    if not _is_int(rq["seed"]):
        return f"malformed:seed is {type(rq['seed']).__name__}"
    if not isinstance(rq["seed_set"], bool):
        return f"malformed:seed_set is {type(rq['seed_set']).__name__}"
    return None


def _join_proposals(sched: list[dict], worker: list[dict]):
    """The event/proposal join: sched seq N's verify events pair with the
    worker seq N-1 proposals that produced them (the documented one-step
    lag; README phase relation 4).

    Returns (joined, n_boundary, missing) where joined is a list of
    (sched_line, producing_worker_line_or_None) for every sched line,
    n_boundary counts the series' FIRST sched line (its producer may sit
    before the scrape window) and missing counts sched lines whose
    producer is absent from the series although it is not the first line
    (a dropped worker record -- a real pairing defect).
    """
    wmap = {w.get("seq"): w for w in worker}
    joined, n_boundary, missing = [], 0, 0
    for i, s in enumerate(sched):
        n = s.get("seq")
        prev = wmap.get(n - 1) if isinstance(n, int) else None
        if prev is not None:
            joined.append((s, prev))
        elif i == 0:
            n_boundary += 1
            joined.append((s, None))
        else:
            missing += 1
            joined.append((s, None))
    return joined, n_boundary, missing


def compare_prefix_chains(base: list[dict], arm: list[dict]) -> dict:
    """Cross-boot committed-prefix-hash identity for one request.

    Each chain is a list of {"ol": int, "phash": str} records (the request's
    committed-prefix state per engine step). The comparison joins on the
    committed OUTPUT length: at every common ol the two boots' whole
    committed token prefix (prompt + output, int64-le sha256) must be
    byte-identical. Returns {"common_ol": n, "mismatches": [...],
    "match": bool} -- match is True iff at least one common ol exists and
    none of them disagree.
    """
    bmap = {r["ol"]: r["phash"] for r in base if _is_int(r.get("ol"))}
    amap = {r["ol"]: r["phash"] for r in arm if _is_int(r.get("ol"))}
    common = sorted(set(bmap) & set(amap))
    mismatches = [{"ol": ol, "base": bmap[ol], "arm": amap[ol]}
                  for ol in common if bmap[ol] != amap[ol]]
    return {"common_ol": len(common), "mismatches": mismatches,
            "match": bool(common) and not mismatches}


def _chains_in_window(recs: list[dict]) -> dict:
    """Per-request committed-prefix chains from a window's sched records.

    Returns {req_id: [{"ol","phash"}, ...]} ordered by seq. Records without
    a phash/ol (old-shape emitters) contribute nothing.
    """
    chains: dict[str, list[dict]] = {}
    for s in sorted(recs, key=lambda r: (r.get("seq") is None,
                                         r.get("seq") or 0)):
        for rq in (s.get("reqs") or []):
            if not isinstance(rq, dict):
                continue
            if _is_int(rq.get("ol")) and isinstance(rq.get("phash"), str):
                chains.setdefault(rq["id"], []).append(
                    {"ol": rq["ol"], "phash": rq["phash"]})
    return chains


def _first_appearance_chains(window_recs: list[dict]) -> dict:
    """The chains of the requests that FIRST commit inside the window.

    The C2 capture protocol runs ONE request per marked window; the window's
    new request is the one whose committed output length starts at 0 inside
    it (a request that decoded before the window carries ol > 0 on every row;
    output lengths never shrink). Returns {req_id: chain} for every such
    request (exactly one under the sequential capture protocol).
    """
    chains = _chains_in_window(window_recs)
    return {rid: ch for rid, ch in chains.items()
            if ch and min(r["ol"] for r in ch) == 0}


def _chain_for_response(win_chains: dict, resp_id: str | None):
    """The window chain of ONE request, attributed by its response id.

    The evidence req ids are `<response id>-<suffix>`; the streamed response
    carries its id on every chunk, so the capture/D5 window picks its OWN
    request's chain instead of the longest chain in the window (the old
    longest-first-appearance pick cross-attributed a previous request's
    bleeding tail lines and saw nothing for single-chunk requests, whose
    first commit record starts at ol>=1). Returns None when the id is
    unknown or no chain matches -- callers fall back to the round-8
    protocol, which keeps the mock world (id-less chunks) on its path.
    """
    if not resp_id:
        return None
    prefix = str(resp_id) + "-"
    matches = [(rid, ch) for rid, ch in win_chains.items()
               if str(rid).startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=lambda kv: (len(kv[1]), kv[0]))[1]


def _c2_semantics_check(series: dict, res: PointResult, cid: str,
                        launch_k: int) -> None:
    """The round-8 C2-field checks (D-014 condition 2).

    Five checks over the window's paired evidence:
      <cid>.fields_worker   every worker line carries the proposal-side
                            fields (branch/nt/pf + per-row temp/seed/
                            seed_set/props), correctly typed;
      <cid>.fields_sched    every sched row carries the verify/commit-side
                            fields (rej_idx/bonus null-able; emit/phash/
                            ol/nct + the sampling mirror), correctly typed;
      <cid>.proposal_join   every sched step with a verify window pairs
                            with the worker step that produced the
                            proposal (the one-step lag);
      <cid>.argmax_greedy   on greedy rows (temp == 0) the committed emit
                            is the drafter's argmax prefix plus the
                            target's own token at the rejection/bonus
                            position (the target-logit/argmax evidence);
      <cid>.prefix_hash     per request the committed-prefix hash chain is
                            well-formed and grows by exactly len(emit)
                            (a regression must re-base the hash).

    Absent fields are MISSING_INSTRUMENT (an emitter predating the
    round-8 contract), wrongly-typed values are MISSING_MALFORMED_
    INSTRUMENT -- never a crash and never a silent pass.
    """
    sched, worker = series["sched"], series["worker"]
    if not sched and not worker:
        return  # no coverage gate here: A1/P2.coverage own the window

    # ---- worker fields ----
    w_absent, w_malformed, w_ok = [], [], 0
    for w in worker:
        problem = _worker_c2_problem(w)
        if problem is None:
            w_ok += 1
        elif problem.startswith("absent:"):
            w_absent.append((w.get("seq"), problem))
        else:
            w_malformed.append((w.get("seq"), problem))
    if w_absent:
        res.add(f"{cid}.fields_worker", False,
                f"{len(w_absent)}/{len(worker)} worker lines lack round-8 "
                f"proposal-side fields {w_absent[:2]} - the emitter "
                "predates the C2 field-completion contract (rebuild on the "
                "iter7f image)", MISSING_INSTRUMENT)
    elif w_malformed:
        res.add(f"{cid}.fields_worker", False,
                f"malformed round-8 worker fields {w_malformed[:2]} - "
                "failing the check instead of crashing the run",
                MISSING_MALFORMED_INSTRUMENT)
    else:
        res.add(f"{cid}.fields_worker", w_ok > 0,
                f"{w_ok} worker lines carry branch/nt/pf + per-row "
                "temp/seed/seed_set/props (typed)")

    # ---- sched fields ----
    s_absent, s_malformed, s_ok = [], [], 0
    for s in sched:
        for rq in (s.get("reqs") or []):
            if not isinstance(rq, dict):
                continue  # the alignment/accounting checks report it
            problem = _sched_c2_problem(rq)
            if problem is None:
                s_ok += 1
            elif problem.startswith("absent:"):
                s_absent.append((s.get("seq"), rq.get("id"), problem))
            else:
                s_malformed.append((s.get("seq"), rq.get("id"), problem))
    if s_absent:
        res.add(f"{cid}.fields_sched", False,
                f"{len(s_absent)} sched rows lack round-8 verify/commit "
                f"fields {s_absent[:2]} - the emitter predates the C2 "
                "field-completion contract", MISSING_INSTRUMENT)
    elif s_malformed:
        res.add(f"{cid}.fields_sched", False,
                f"malformed round-8 sched fields {s_malformed[:2]}",
                MISSING_MALFORMED_INSTRUMENT)
    else:
        res.add(f"{cid}.fields_sched", s_ok > 0,
                f"{s_ok} sched rows carry rej_idx/bonus + emit/phash/ol/nct "
                "+ the sampling mirror (typed)")

    # ---- proposal join + argmax evidence + hash chain ----
    joined, n_boundary, n_missing = _join_proposals(sched, worker)
    joined_n = len(joined) - n_boundary
    res.add(f"{cid}.proposal_join",
            n_missing == 0 and joined_n > 0,
            f"sched steps {len(sched)}: proposal joins {joined_n} "
            "(producer = the worker step one seq earlier), boundary skips "
            f"{n_boundary} (the series' first sched line), MISSING producers "
            f"{n_missing}")
    arg_checked = arg_bad = prob_checked = skipped_props = 0
    stop_trimmed = length_capped = 0
    first_bad = None
    for s, w_prev in joined:
        if w_prev is None:
            continue  # the series' first sched line: no producer in the window
        if _reqs_field_problem(w_prev) is not None:
            # the producer line's reqs payload is absent/malformed -- the
            # alignment check names it; the argmax check must not pass
            # vacuously over the unjoinable verify rows
            skipped_props += 1
            continue
        prev_props = {rq.get("id"): rq.get("props")
                      for rq in w_prev["reqs"] if isinstance(rq, dict)}
        for rq in (s.get("reqs") or []):
            if not isinstance(rq, dict):
                continue
            win = rq.get("win") or 0
            if not win or rq.get("rej_idx") is None:
                continue  # no verify window this step
            props = prev_props.get(rq.get("id"))
            emit, bonus = rq.get("emit"), rq.get("bonus")
            rej_idx = rq.get("rej_idx")
            if not isinstance(props, list) or not isinstance(emit, list):
                skipped_props += 1  # the fields checks report the cause
                continue
            prob_checked += 1
            temp = rq.get("temp")
            greedy = isinstance(temp, (int, float)) and temp == 0
            if greedy:
                arg_checked += 1
                bad = None
                if not _is_int(rej_idx) or rej_idx < 0 or rej_idx > win:
                    bad = f"rej_idx {rej_idx} outside [0, win={win}]"
                elif len(emit) < rej_idx + 1:
                    # STOP-TRIM (first live firing, record run 2026-10-15):
                    # the request TERMINATED inside the accepted prefix -- a
                    # committed member of it was the EOS/stop token, the
                    # engine stopped the output there, and the verify
                    # accounting still counts the positions it compared
                    # (rej_idx=3 while the output list received only the
                    # stop token). The tail assertions lose their subject;
                    # the argmax evidence still covers every COMMITTED
                    # token: emit must BE the accepted argmax prefix.
                    #
                    # Termination must be PROVED, not presumed (the astra
                    # hold of 2026-10-15):
                    # the commit record carries the scheduler's stopped
                    # flag -- _update_request_with_output sets it whenever
                    # the append itself fired a stop (the EOS/stop token
                    # and the max_tokens length cap alike, each with the
                    # same trim). A shortened prefix is accepted ONLY on
                    # stopped=True; a MISSING flag, a False one, or an
                    # empty emit without that proof is an explicit failure
                    # -- never a silent pass: without the flag a truncated
                    # or mis-scraped emit would be indistinguishable from
                    # a real stop-trim, and an empty one would erase the
                    # window's evidence. An optional stop/finish marker
                    # (stop_reason, else finish_reason) only labels the
                    # trim in the report: the length cap stops the output
                    # the same way and earns the same flag, so it passes
                    # on the same proof and is counted apart when the
                    # marker says so.
                    stopped = rq.get("stopped")
                    if stopped is not True:
                        if "stopped" not in rq:
                            why = ("the commit record carries no stopped "
                                   "flag (termination unproven; the "
                                   "emitter predates the stopped field)")
                        else:
                            why = (f"stopped={stopped!r} is not true "
                                   "(termination unproven)")
                        bad = (f"stop-trimmed emit {emit} (rej_idx "
                               f"{rej_idx} counts {rej_idx + 1} committed "
                               f"positions) but {why} - a shortened "
                               "matching prefix is accepted only on "
                               "stopped=True")
                    elif emit != props[:len(emit)]:
                        bad = (f"stop-trimmed emit {emit} is not the "
                               f"accepted argmax prefix props[:{len(emit)}]"
                               f"={props[:len(emit)]} (the committed tokens "
                               "must still be the drafter's argmax prefix)")
                    else:
                        stop_trimmed += 1
                        # "or" (not .get default): a present-but-null
                        # stop_reason must still fall through to
                        # finish_reason; and the enum names the emitter
                        # records (FINISHED_LENGTH_CAPPED) contain the
                        # substring rather than the prefix
                        marker = rq.get("stop_reason") or rq.get("finish_reason")
                        if isinstance(marker, str) \
                                and "length" in marker.lower():
                            length_capped += 1
                elif emit[:rej_idx] != props[:rej_idx]:
                    bad = (f"accepted prefix emit[:{rej_idx}]"
                           f"={emit[:rej_idx]} != props[:{rej_idx}]"
                           f"={props[:rej_idx]} (the greedy verify must "
                           "accept exactly the drafter's argmax prefix)")
                elif len(emit) != rej_idx + 1:
                    bad = (f"emit length {len(emit)} != rej_idx+1 "
                           f"{rej_idx + 1} (single-sample verify)")
                elif not _is_int(bonus) or (emit and emit[-1] != bonus):
                    bad = f"emit tail != bonus {bonus!r}"
                elif rej_idx < len(props) and bonus == props[rej_idx]:
                    bad = ("bonus == the rejected draft token (a true "
                           "rejection must resample a DIFFERENT token)")
                if bad and first_bad is None:
                    first_bad = {"seq": s.get("seq"), "id": rq.get("id"),
                                 "why": bad}
                arg_bad += 1 if bad else 0
    if prob_checked == 0 and skipped_props:
        # verify rows existed but no proposal side was joinable: the emitter
        # predates the round-8 worker fields (the fields check above names
        # them) -- MISSING_INSTRUMENT, never a vacuous pass or a FAIL lie
        res.add(f"{cid}.argmax_greedy", False,
                f"{skipped_props} verify rows had no joinable proposal side "
                "(the worker lines lack the round-8 props fields)",
                MISSING_INSTRUMENT)
    else:
        res.add(f"{cid}.argmax_greedy",
                prob_checked > 0 and arg_bad == 0,
                f"verify rows {prob_checked} (greedy rows checked "
                f"{arg_checked}, probabilistic rows exempt; stop-trimmed "
                f"{stop_trimmed}"
                + (f", length-capped {length_capped}" if length_capped
                   else "")
                + "): emit == the accepted argmax prefix + the "
                "target's token at the rejection/bonus position"
                + (f"; offenders: {first_bad}" if first_bad else ""))

    # ---- per-request committed-prefix hash chain ----
    per_req: dict[str, list[tuple]] = {}
    for s in sched:
        for rq in (s.get("reqs") or []):
            if isinstance(rq, dict) and rq.get("phash") is not None:
                per_req.setdefault(rq.get("id"), []).append(
                    (s.get("seq"), rq))
    hash_bad = []
    for rid, rows in per_req.items():
        rows.sort(key=lambda t: (t[0] is None, t[0] or 0))
        prev_ol = None
        prev_phash = None
        for seq, rq in rows:
            ol, ph, emit = rq.get("ol"), rq.get("phash"), rq.get("emit") or []
            if not _is_int(ol) or not isinstance(ph, str) or len(ph) != 64:
                hash_bad.append({"id": rid, "seq": seq,
                                 "why": "malformed phash/ol"})
                break
            if prev_ol is not None:
                grew = ol - prev_ol
                if grew != len(emit):
                    hash_bad.append({
                        "id": rid, "seq": seq,
                        "why": f"ol grew {grew} != len(emit) {len(emit)}"})
                    break
                if grew == 0 and ph == prev_phash:
                    pass  # nothing committed: the prefix is unchanged
                elif grew > 0 and ph == prev_phash:
                    hash_bad.append({
                        "id": rid, "seq": seq,
                        "why": f"ol grew {grew} but the prefix hash did not "
                               "move (stale hash)"})
                    break
            prev_ol, prev_phash = ol, ph
    res.add(f"{cid}.prefix_hash",
            len(per_req) > 0 and not hash_bad,
            f"{len(per_req)} tracked requests; chains grow by exactly "
            "len(emit) with the hash moving on growth (regressions re-base)"
            + (f"; offenders: {hash_bad[:2]}" if hash_bad else ""))


# ---------------------------------------------------------------------------
# AWFP boot fingerprints (round 8, D-014 condition 1)
# ---------------------------------------------------------------------------
def parse_awfp(lines: list[tuple[float, str]]) -> list[dict]:
    """Parse the AWFP fingerprint lines (tag + one JSON object)."""
    recs = []
    for ts, line in lines:
        idx = line.find(TAG_AWFP + " ")
        if idx < 0:
            continue
        payload = line[idx + len(TAG_AWFP) + 1:]
        start = payload.find("{")
        if start < 0:
            continue
        try:
            rec = json.loads(payload[start:])
        except json.JSONDecodeError:
            recs.append({"_bad": True, "ts": ts})
            continue
        rec.setdefault("ts", ts)
        recs.append(rec)
    return recs


def awfp_failures(lines: list[tuple[float, str]]) -> list[str]:
    """The AWFP-FAIL lines (plain text, one per fingerprint stage failure)."""
    return [line for _, line in lines if "AWFP-FAIL" in line]


def awfp_policy_by_boot() -> dict:
    """The policy_sha collected per boot name this run (cross-boot compare)."""
    return dict(AWFP_POLICY_BY_BOOT)


AWFP_POLICY_BY_BOOT: dict[str, str] = {}


AWFP_POLICY_STEP_BY_BOOT: dict[str, int] = {}


def _awfp_collect_policy(logs, boot_ts: float, key: str) -> str | None:
    """Collect the LAST AWFP-POLICY record of one boot into the registry.

    The record's scheduled-step number is kept alongside the sha (the
    reviewer's provenance finding: a sha without its step cannot be told
    apart from an earlier, unsettled one). A FAILED collection must NOT
    overwrite the registry: run_p4's early/settled drift check reads the
    registry BEFORE calling this, and an overwrite-on-failure would erase
    the very value the check needs."""
    try:
        lines = logs.since(boot_ts)
    except Exception:  # noqa: BLE001
        return None
    polys = [r for r in parse_awfp(lines) if r.get("kind") == "POLICY"
             and isinstance(r.get("policy_sha"), str)]
    if not polys:
        return None
    sha = polys[-1]["policy_sha"]
    step = polys[-1].get("step")
    AWFP_POLICY_BY_BOOT[key] = sha
    if isinstance(step, int):
        AWFP_POLICY_STEP_BY_BOOT[key] = step
    else:
        # a fresh record with no parseable step must not inherit the
        # previous collection's step (the reviewer's stale-provenance
        # defect: the sha updated while the step stamp stayed behind)
        AWFP_POLICY_STEP_BY_BOOT.pop(key, None)
    return sha


def _awfp_boot_check(ctx, res: PointResult, point: str) -> bool:
    """The boot's runtime fingerprints exist AND prove the two fixes.

    Three checks per point: the CODE record (the marlin module's own
    load-time state: atomic_add call sites + shim wiring), the PROBE
    record (the behavioral canonical-order proof through the patched
    path + shim env gate on), and no AWFP-FAIL line anywhere. A boot of
    a pre-iter7f image reports MISSING_INSTRUMENT here -- the fingerprint
    layer's absence is itself loud, never a silent pass.
    """
    lines = [l for _, l in ctx.logs.since(ctx.boot_ts)]
    recs = parse_awfp(ctx.logs.since(ctx.boot_ts))
    fails = awfp_failures(ctx.logs.since(ctx.boot_ts))
    code = [r for r in recs if r.get("kind") == "CODE"]
    probe = [r for r in recs if r.get("kind") == "PROBE"]
    ok = True
    if not code:
        res.add(f"{point}.FP.code_state", False,
                f"no {TAG_AWFP} CODE record since boot - this image does "
                "not carry the runtime fingerprint layer (rebuild on "
                "iter7f or later)", MISSING_INSTRUMENT)
        ok = False
    else:
        bad = [r for r in code if r.get("ok") is not True]
        res.add(f"{point}.FP.code_state", not bad,
                f"{len(code)} CODE record(s) since boot; atomic_add call "
                "sites hard-set false + stable-sort shim wiring verified"
                + (f"; offenders: {[r.get('file') or r for r in bad[:2]]}"
                   if bad else ""),
                FAIL if bad else PASS)
        ok &= not bad
    if not probe:
        res.add(f"{point}.FP.probe", False,
                f"no {TAG_AWFP} PROBE record since boot - the behavioral "
                "shim/atomic_add probe never ran (check the worker "
                "process's load_model tail)", MISSING_INSTRUMENT)
        ok = False
    else:
        not_passed = [r for r in probe if r.get("passed") is not True]
        res.add(f"{point}.FP.probe", not not_passed,
                f"{len(probe)} PROBE record(s): canonical-order proof "
                "(direct + the real align composition) and the "
                "atomic_add call-site assertion"
                + (f"; failed probes: {not_passed[:1]}" if not_passed else ""),
                FAIL if not_passed else PASS)
        ok &= not not_passed
    if fails:
        res.add(f"{point}.FP.failures", False,
                f"{len(fails)} AWFP-FAIL line(s) since boot: {fails[:2]}",
                FAIL)
        ok = False
    else:
        res.add(f"{point}.FP.failures", True,
                f"no AWFP-FAIL line since boot ({len(recs)} AWFP records "
                "parsed)")
    return ok


# ---------------------------------------------------------------------------
# point results
# ---------------------------------------------------------------------------
class Check:
    def __init__(self, cid: str, ok: bool, detail: str, status: str = PASS):
        self.id, self.ok, self.detail, self.status = cid, ok, detail, status

    def to_json(self):
        return {"id": self.id, "status": self.status, "detail": self.detail}


class PointResult:
    def __init__(self, point: str):
        self.point = point
        self.checks: list[Check] = []
        self.data: dict = {}

    def add(self, cid, ok, detail, status=None):
        st = status or (PASS if ok else FAIL)
        self.checks.append(Check(cid, bool(ok), detail, st))
        return self.checks[-1]

    def check(self, cid) -> Check | None:
        return next((c for c in self.checks if c.id == cid), None)

    @property
    def status(self) -> str:
        sts = [c.status for c in self.checks]
        if any(s == FAIL for s in sts):
            return FAIL
        if any(s in (MISSING_INSTRUMENT, MISSING_MALFORMED_INSTRUMENT)
               for s in sts):
            # the malformed class aggregates as MISSING_INSTRUMENT at point
            # level (both are FAIL-class instrumentation failures); the
            # distinction stays visible on the check itself
            return MISSING_INSTRUMENT
        if any(s == WARN for s in sts):
            return WARN
        if not sts:
            return SKIP
        return PASS

    def to_json(self):
        return {"point": self.point, "status": self.status,
                "checks": [c.to_json() for c in self.checks], "data": self.data}


# ---------------------------------------------------------------------------
# boot management
# ---------------------------------------------------------------------------
class BootSpec:
    """One gate boot: a scratch copy of a recipe config dir with rewrites.

    base_config_rel (round 6): the repo-relative config dir this boot COPIES,
    declared on the spec itself. The round-5 auto-baseline hardcoded the
    gate's base_config_dir (v18a-g2, whose committed env sets
    ADAPTIVE_WIDTH_MODE='armed'/ADAPTIVE_WIDTH_FORCE='oscillate') as the copy
    source for the off/MTP=3 floor boot, so the P4 baseline would have been
    silently captured from an ARMED boot. The source now lives on the spec
    (make_boot_specs pins baseline -> baseline_config_dir/v16b, gate boots ->
    base_config_dir) and prepare_config_dir refuses a spec without one --
    exactly one source of truth, never a silent default.

    sync_scheduling (round 5): every harness-driven boot appends
    --no-async-scheduling to the copied env's EXTRA. The D-009 contract
    excludes async scheduling pending verification, but the round-4 gate run
    silently RAN the AsyncScheduler: SchedulerConfig.async_scheduling=None
    auto-resolves to True for MTP boots ("mtp" is an EagleModelTypes member,
    the executor supports async, no incompatibility fired) and the argv-only
    P3.S7 grep could not see it. patch_serve (below) is the AWCFG/SPECDBG
    serve.sh insert -- it requires the v17a+ AW anchor block, which the v16b
    baseline copy lacks, so the baseline boot sets it False (explicitly
    logged, never silently skipped).
    """

    def __init__(self, name, mtp, aw_mode, aw_force, launch_k, candidates,
                 evidence=1, image_override=None, extra_env=None,
                 sync_scheduling=True, patch_serve=True, base_config_rel=None):
        self.name = name
        self.mtp = int(mtp)
        self.aw_mode = aw_mode            # off|shadow|armed|force
        self.aw_force = aw_force          # None|'oscillate'|'3'|...
        self.launch_k = int(launch_k)
        self.candidates = sorted(set(int(c) for c in candidates))
        self.evidence = int(evidence)
        self.image_override = image_override
        self.extra_env = dict(extra_env or {})
        self.sync_scheduling = bool(sync_scheduling)
        self.patch_serve = bool(patch_serve)
        self.base_config_rel = base_config_rel

    def env_overrides(self) -> dict:
        ov = {"MTP": str(self.mtp)}
        if self.aw_mode and self.aw_mode != "off":
            ov["ADAPTIVE_WIDTH_MODE"] = self.aw_mode
            if self.aw_force:
                ov["ADAPTIVE_WIDTH_FORCE"] = str(self.aw_force)
        else:
            # Round 6: pin the OFF floor explicitly (and empty the force
            # knob, which serve.sh's -n test treats as unset) so an
            # armed/oscillate line inherited from the copied env can never
            # leak through, whatever dir the spec copies. The round-5
            # review: env_overrides(aw_mode=None) did not clear or override
            # the mode.
            ov["ADAPTIVE_WIDTH_MODE"] = "off"
            ov["ADAPTIVE_WIDTH_FORCE"] = ""
        ov["SPECDBG_EVIDENCE"] = str(self.evidence)
        ov.update(self.extra_env)
        return ov


def _rewrite_env(env_path: Path, overrides: dict) -> list[str]:
    """Assert-guarded env rewrite (key='value' lines)."""
    lines = env_path.read_text().splitlines()
    actions = []
    for key, value in overrides.items():
        pat = re.compile(rf"^{re.escape(key)}=.*$")
        hits = [i for i, l in enumerate(lines) if pat.match(l)]
        new = f"{key}='{value}'"
        if len(hits) == 1:
            old = lines[hits[0]]
            if old != new:
                lines[hits[0]] = new
                actions.append(f"{key}:set({old} -> {new})")
            else:
                actions.append(f"{key}:unchanged")
        elif len(hits) == 0:
            lines.append(new)
            actions.append(f"{key}:appended")
        else:
            raise AssertionError(
                f"env rewrite guard failed: key {key} matches {len(hits)} lines")
    env_path.write_text("\n".join(lines) + "\n")
    return actions


_SERVE_ANCHOR = (
    '  [ -n "${ADAPTIVE_WIDTH_FORCE:-}" ] && '
    'AW_ARGS+=(-e "VLLM_ADAPTIVE_WIDTH_FORCE=$ADAPTIVE_WIDTH_FORCE")\n'
    "fi\n"
)
_SERVE_INSERT = (
    "# ---- gate-stage1b additions (inserted by scripts/gate-stage1b; idempotent) ----\n"
    "AWCFG_ARGS=()\n"
    'if [ -n "${ADAPTIVE_WIDTH_CONFIG_FILE:-}" ]; then\n'
    '  AWCFG_ARGS+=(-v "${ADAPTIVE_WIDTH_CONFIG_FILE}:/aw-gate-config.json:ro")\n'
    '  AWCFG_ARGS+=(-e "VLLM_ADAPTIVE_WIDTH_CONFIG=/aw-gate-config.json")\n'
    "fi\n"
    "SPECDBG_ARGS=()\n"
    'if [ -n "${SPECDBG_EVIDENCE:-}" ]; then\n'
    '  SPECDBG_ARGS+=(-e "VLLM_SPECDBG_EVIDENCE=${SPECDBG_EVIDENCE}")\n'
    "fi\n"
    "# ---- end gate-stage1b additions ----\n"
)
_SERVE_RUN_ANCHOR = '  "${AW_ARGS[@]}" \\\n'
_SERVE_RUN_INSERT = '  "${AWCFG_ARGS[@]}" \\\n  "${SPECDBG_ARGS[@]}" \\\n'


def _patch_serve_sh(serve_path: Path) -> list[str]:
    """Idempotent, assert-guarded serve.sh insert for the gate env plumbing."""
    s = serve_path.read_text()
    if "AWCFG_ARGS" in s:
        return ["serve.sh:already-patched"]
    n1 = s.count(_SERVE_ANCHOR)
    if n1 != 1:
        raise AssertionError(
            f"serve.sh patch guard failed: AW-block anchor count {n1} in {serve_path}")
    s = s.replace(_SERVE_ANCHOR, _SERVE_ANCHOR + _SERVE_INSERT, 1)
    n2 = s.count(_SERVE_RUN_ANCHOR)
    if n2 != 1:
        raise AssertionError(
            f"serve.sh patch guard failed: DOCKER_RUN anchor count {n2} in {serve_path}")
    s = s.replace(_SERVE_RUN_ANCHOR, _SERVE_RUN_ANCHOR + _SERVE_RUN_INSERT, 1)
    # Round 7: syntax-guard the PATCHED text, not the file on disk. The old
    # form ran `bash -n <serve_path>` BEFORE the write - which validated the
    # UNPATCHED original and could never catch a syntax error the insert
    # itself introduced (the guard's stated intent). The patched text now
    # goes to a temp file next to the target, is bash -n checked there, and
    # only then replaces the original.
    tmp_syntax = serve_path.with_name(serve_path.name + ".gatesyntax")
    tmp_syntax.write_text(s)
    try:
        chk = subprocess.run(["bash", "-n", str(tmp_syntax)],
                             capture_output=True, timeout=30)
        if chk.returncode != 0:
            raise AssertionError(
                f"serve.sh patch produced invalid bash syntax: "
                f"{chk.stderr.decode()[:300]}")
    finally:
        tmp_syntax.unlink(missing_ok=True)
    serve_path.write_text(s)
    return ["serve.sh:patched(AWCFG_ARGS,SPECDBG_ARGS)"]


def _write_aw_config(path: Path, candidates: list[int], launch_k: int) -> None:
    """Adaptive-width config for the gate: candidates bounded by launch K in
    BOTH the top-level config and EVERY tier (D-009 trap 1: install() checks
    only the top-level max, but slot_for() merges tier candidates, so an
    uncorrected tier would silently create slots above launch capacity)."""
    if max(candidates) > launch_k:
        raise AssertionError(f"gate candidates {candidates} exceed launch_k {launch_k}")
    cfg = {
        "candidate_steps": list(candidates),
        "tiers": {"1": {"candidate_steps": list(candidates)}},
        "ema_alpha": 0.2,
        "update_interval": 5,
        "warmup_batches": 10,
        "down_hysteresis": -0.25,
        "up_hysteresis": 0.0,
        "ceiling_coeff": 0,
    }
    path.write_text(json.dumps(cfg, indent=1) + "\n")


def _append_extra_arg(env_path: Path, flag: str) -> str:
    """Append one CLI flag to the env's EXTRA value (assert-guarded).

    EXTRA is the recipe's passthrough into the vllm argv (both v16b and
    v18a-g2 serve.sh expand it inside the DOCKER_RUN array), so appending
    here reaches the engine argv without another serve.sh anchor. The
    rewrite asserts exactly one EXTRA line and refuses to double-append.
    """
    text = env_path.read_text()
    hits = re.findall(r"^EXTRA='(.*)'$", text, re.M)
    if len(hits) != 1:
        raise AssertionError(
            f"env EXTRA guard failed: {len(hits)} EXTRA lines in {env_path} "
            "(want exactly one single-quoted value)")
    old = hits[0]
    if flag in old:
        return f"EXTRA:{flag}:already-present"
    new = f"{old} {flag}".strip()
    _rewrite_env(env_path, {"EXTRA": new})
    return f"EXTRA:append({flag})"


def prepare_config_dir(spec: BootSpec, scratch_root: Path,
                       repo_root: Path) -> Path:
    """Copy the spec's OWN base config dir and apply guarded rewrites. Never
    touches the repository original.

    Round 6: the copy source is spec.base_config_rel -- the round-5
    auto-baseline hardcoded the gate's base_config_dir (v18a-g2, whose
    committed env sets ADAPTIVE_WIDTH_MODE='armed'/'FORCE=oscillate') as the
    copy source for the off/MTP=3 floor boot, silently arming the P4
    baseline. The source now lives on the BootSpec (make_boot_specs pins
    baseline -> baseline_config_dir/v16b, gate boots -> base_config_dir), so
    there is exactly one source of truth and a spec without one is an
    assert-guard refusal, never a silent default."""
    base_rel = spec.base_config_rel
    if not base_rel:
        raise AssertionError(
            f"BootSpec {spec.name} carries no base_config_rel - every boot "
            "must declare the config dir it copies (round 6: the baseline "
            "boot copied base_config_dir (armed v18a-g2) instead of "
            "baseline_config_dir (v16b) while the source was hardcoded)")
    dst = scratch_root / f"config-{spec.name}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(repo_root) / base_rel, dst)
    log = [f"base_config_dir:{base_rel}"]
    log += _rewrite_env(dst / "env", spec.env_overrides())
    aw_json = dst / "aw-gate-config.json"
    _write_aw_config(aw_json, spec.candidates, spec.launch_k)
    # ADAPTIVE_WIDTH_CONFIG (a host path) must NOT be set in env: the
    # in-container path comes from ADAPTIVE_WIDTH_CONFIG_FILE via the serve.sh
    # insert below.
    env_text = (dst / "env").read_text()
    if re.search(r"^ADAPTIVE_WIDTH_CONFIG=", env_text, re.M):
        raise AssertionError(
            "generated env must not set ADAPTIVE_WIDTH_CONFIG (a host path is "
            "invisible in-container); use ADAPTIVE_WIDTH_CONFIG_FILE")
    if spec.patch_serve:
        log += _rewrite_env(dst / "env",
                            {"ADAPTIVE_WIDTH_CONFIG_FILE": str(aw_json)})
    else:
        # Explicit skip, never silent: nothing mounts aw-gate-config.json on
        # an unpatched serve.sh, so the env line would be dead config (the
        # round-5 review flagged it as "written but never mounted" on the
        # baseline copy).
        log.append("env:ADAPTIVE_WIDTH_CONFIG_FILE:skipped(patch_serve=False)")
    # Round 5: the documented gate path is the SYNC scheduler (D-009 excludes
    # async pending verification). The scheduler config auto-resolves
    # async_scheduling=None to True for MTP boots, so the exclusion must be
    # EXPLICIT or the gate silently exercises the wrong path.
    if spec.sync_scheduling:
        log.append(_append_extra_arg(dst / "env", NO_ASYNC_FLAG))
    if spec.patch_serve:
        log += _patch_serve_sh(dst / "serve.sh")
    else:
        # Explicit skip, never silent: the baseline (v16b) serve.sh predates
        # the AW anchor block the AWCFG/SPECDBG insert anchors on, and the
        # off-floor boot needs neither insert (no controller, no emitters on
        # the baseline image).
        log.append("serve.sh:skipped(patch_serve=False; off-floor boot needs "
                   "no AWCFG/SPECDBG plumbing)")
    if spec.image_override:
        (dst / "image").write_text(spec.image_override + "\n")
        log.append(f"image:set({spec.image_override})")
    (dst / "prepare.log").write_text("\n".join(log) + "\n")
    return dst


def launch_config_dir(config_dir: Path, log_path: Path) -> None:
    """Boot by running the copied config dir's launch.sh --run (boot-matrix
    precedent). Replaces the running container (docker rm -f + run in serve.sh)."""
    with open(log_path, "w") as lf:
        subprocess.run(["bash", "launch.sh", "--run"], cwd=str(config_dir),
                       stdout=lf, stderr=subprocess.STDOUT, timeout=1200, check=True)


def wait_ready(base: str, timeout_s: float, poll_s: float = 10.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"server not ready after {timeout_s}s")


def container_argv(container: str) -> list[str]:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{json .Args}}", container],
        capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout.strip())


def container_env(container: str) -> dict:
    """The container's Config.Env as a dict (docker inspect, read-only).

    The round-1 review landed an attack here: the old baseline-boot check
    grepped only the container ARGV for VLLM_ADAPTIVE_WIDTH_MODE, but the
    wiring's env travels in Config.Env (docker run -e), so a shadow boot
    validated as the 'off floor'. Env is the authoritative source; argv is
    kept as a secondary signal for launchers that inline it."""
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{json .Config.Env}}", container],
        capture_output=True, text=True, timeout=30, check=True)
    env = {}
    for entry in json.loads(out.stdout.strip()):
        if "=" in entry:
            k, _, v = entry.partition("=")
            env[k] = v
    return env


def container_image(container: str) -> str:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.Config.Image}}", container],
        capture_output=True, text=True, timeout=30, check=True)
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# scenario prompts (P3 / P4 workloads; probe_determinism patterns)
# ---------------------------------------------------------------------------
def long_ctx(target_tokens: int, needle_id: str) -> str:
    filler = (
        "The maintenance log for line {i:03d} records routine checks on the conveyor "
        "bearing housing, the filter differential pressure, and the gearbox oil level; "
        "no anomalies were observed during the morning shift on any of these items. "
    )
    needle = (
        f"NOTE {needle_id}: the auth token for the staging deploy is "
        f"AX-{needle_id}-7Q4F-2026 and the rollback owner is Dana R."
    )
    reps = max(1, target_tokens // 60)
    parts = []
    cut = int(reps * 0.4)
    for i in range(reps):
        if i == cut:
            parts.append(needle + "\n")
        parts.append(filler.format(i=i))
    return "".join(parts)


P4_PROMPTS = [
    ("short-code", "Write a Python function that returns the median of three numbers."),
    ("short-zh", "用三句话解释什么是快速排序。"),
    ("needle-8k", long_ctx(8000, "P4G8") + "\n\nQuote the NOTE line exactly."),
    ("copy-exact", "Repeat exactly, character for character: alpha-beta-gamma-delta-epsilon."),
    ("random-reject", "Output 30 random 8-letter gibberish words, comma separated."),
]


# ---------------------------------------------------------------------------
# S4 calibration helpers (round 7): the trimmed-window trigger needs a step
# whose token-budget leftover is comparable to the decode rows' pending
# window demand, and the only harness-visible knob for that leftover is the
# long prompt's tokenized length (its single prefill chunk leaves
# `--max-num-batched-tokens - prompt_tokens` tokens for the rest of the
# step). The length is MEASURED from the server's own usage, never estimated:
# the round-5 README's chars-per-token guess was ~13% off the live S6
# measurement (3000 target -> 2203 prompt_tokens).
# ---------------------------------------------------------------------------
_S4_FILLER = (
    "The maintenance log for line {i:03d} records routine checks on the conveyor "
    "bearing housing, the filter differential pressure, and the gearbox oil level; "
    "no anomalies were observed during the morning shift on any of these items. "
)
_S4_TAIL_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu one two three four five six seven eight nine ten eleven "
    "twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "north south east west river mountain valley forest meadow harbor island "
    "copper iron silver marble granite basalt amber coral jade pearl".split()
)


def s4_prompt(reps: int, salt: int, tail_words: int) -> str:
    """The S4 calibration prompt: `reps` copies of one filler sentence plus a
    word tail (each tail word is assumed ~one token; the candidate is
    measured and the tail adjusted onto the exact target, so the assumption
    only has to hold to +-1). The SALT digit rotates in the FIRST sentence
    only: same character count - and, for digit BPE tokens, the same token
    count - so the measured length carries across variants, while the token
    stream diverges within the first sentence and the prefix cache cannot
    absorb a repeated attempt's prefill (a fully cached chunk would consume
    ~no token budget and could never trim anything)."""
    first = _S4_FILLER.format(i=salt % 1000)
    body = first + "".join(_S4_FILLER.format(i=i) for i in range(1, max(1, reps)))
    tail = " ".join(_S4_TAIL_WORDS[j % len(_S4_TAIL_WORDS)]
                    for j in range(max(0, tail_words)))
    return (body + " " + tail).strip()


def _s4_fallback(p3cfg: dict) -> str:
    """The historical static shape (used only when the server's usage cannot
    drive the calibration; logged as such, never silently assumed calibrated)."""
    return long_ctx(int(p3cfg.get("s4_fallback_target_tokens", 9000)), "S4TR") \
        + "\n\nOne line summary."


def _s4_shape(ctx: GateContext, p3cfg: dict) -> dict:
    """Build the retuned S4 attempts (round 7): up to three long prompts
    calibrated onto neighbouring budget-leftover targets (covering the
    possible per-row demand constants), plus the shape notes the check detail
    reports so a live run explains its own outcome."""
    budget = int(p3cfg.get("s4_max_batched_tokens", 8192))
    targets = [int(x) for x in p3cfg.get("s4_leftover_targets", [8, 7, 11])]
    notes: dict = {"budget": budget, "leftover_targets": targets,
                   "calibrated": False}
    fallback = {"attempts": [_s4_fallback(p3cfg)], "notes": notes}
    if not targets:
        notes["note"] = "no leftover targets configured; static fallback"
        return fallback

    def _ptoks(resp) -> int | None:
        u = (resp or {}).get("usage") or {}
        v = u.get("prompt_tokens")
        return int(v) if isinstance(v, (int, float)) else None

    try:
        m1 = _ptoks(ctx.server.chat([{"role": "user",
                                      "content": s4_prompt(40, 1, 0)}],
                                    max_tokens=4, temperature=0.0))
        m2 = _ptoks(ctx.server.chat([{"role": "user",
                                      "content": s4_prompt(80, 2, 0)}],
                                    max_tokens=4, temperature=0.0))
        if m1 is None or m2 is None or m2 <= m1:
            notes["note"] = (f"usage probes unusable (m1={m1}, m2={m2}); "
                             "static fallback")
            return fallback
        per_rep = (m2 - m1) / 40.0
        base = m1 - 40.0 * per_rep  # template + fixed overhead, tokens
        mid = targets[0]
        target_len = budget - mid
        reps = max(1, int((target_len - base) // per_rep))
        residual = target_len - (base + per_rep * reps)
        tail0 = max(0, int(round(residual)))
        cand = s4_prompt(reps, 3, tail0)
        m3 = _ptoks(ctx.server.chat([{"role": "user", "content": cand}],
                                    max_tokens=4, temperature=0.0))
        if m3 is None:
            notes["note"] = "candidate measurement failed; static fallback"
            return fallback
        attempts = []
        planned = {}
        for k, left in enumerate(targets):
            # exact residual correction: the candidate measured m3 tokens at
            # tail0; adding/removing (budget-left) - m3 tail words lands the
            # length on budget-left (the word-token assumption is +-1).
            delta = (budget - left) - m3
            attempts.append(s4_prompt(reps, 10 + k, tail0 + delta))
            planned[str(10 + k)] = m3 + delta
        notes.update({"calibrated": True, "probe_tokens": [m1, m2, m3],
                      "tokens_per_rep": round(per_rep, 3), "reps": reps,
                      "tail_words": tail0, "planned_prompt_tokens": planned})
        return {"attempts": attempts, "notes": notes}
    except Exception as e:  # noqa: BLE001
        notes["note"] = f"calibration error ({e}); static fallback"
        return fallback


# ---------------------------------------------------------------------------
# gate context
# ---------------------------------------------------------------------------
class GateContext:
    def __init__(self, cfg: dict, metrics: MetricsSource, logs: LogSource,
                 server: ServerClient, out_dir: Path, boot_ts: float,
                 launch_k: int, argv: list[str] | None = None,
                 install_log_ok: bool | None = None,
                 boot_name: str | None = None):
        self.cfg = cfg
        self.metrics = metrics
        self.logs = logs
        self.server = server
        self.out_dir = out_dir
        self.boot_ts = boot_ts
        self.launch_k = int(launch_k)
        self.argv = argv if argv is not None else []
        self.install_log_ok = install_log_ok
        self.boot_name = boot_name
        self._pair_prev: dict | None = None

    def now(self) -> float:
        return self.metrics.now()

    def sleep(self, s: float) -> None:
        self.metrics.sleep(s)

    def evidence(self, since: float | None = None, until: float | None = None) -> list[dict]:
        recs = parse_evidence(self.logs.since(self.boot_ts if since is None else since))
        if until is not None:
            recs = [r for r in recs if r.get("ts", 0) <= until]
        return recs

    def pair_mark(self) -> None:
        """Snapshot the counters before a scenario window."""
        self._pair_prev = self.metrics.fetch()

    def pair_prev(self) -> dict:
        return self._pair_prev or {}


# ---------------------------------------------------------------------------
# workload drivers
# ---------------------------------------------------------------------------
def run_steady(server: ServerClient, n: int, max_tokens: int,
               sleep=time.sleep) -> list[dict]:
    """Short concurrent decode load."""
    outs = []

    def one(i: int):
        try:
            server.chat([{"role": "user",
                          "content": f"Count from {i} to {i + 20}, comma separated."}],
                        max_tokens=max_tokens, temperature=0.0)
            outs.append({"i": i, "ok": True})
        except Exception as e:  # noqa: BLE001
            outs.append({"i": i, "ok": False, "error": str(e)})

    threads = []
    for i in range(n):
        t = threading.Thread(target=one, args=(i,))
        t.start()
        threads.append(t)
        sleep(0.4)
    for t in threads:
        t.join()
    return outs


def run_waves(server: ServerClient, cfg_wave: dict, sleep=time.sleep) -> list[dict]:
    """Staggered overlapping waves with different lifetimes so the running set
    reorders between proposal-width flips (P2 requirement)."""
    sizes = cfg_wave["p2_max_tokens"]
    outs = []

    def one(wave: int, slot: int):
        try:
            mt = sizes[slot % len(sizes)]
            server.chat([{"role": "user",
                          "content": f"Wave {wave} slot {slot}: list {mt // 2} small primes."}],
                        max_tokens=mt, temperature=0.0, seed=1000 + wave * 10 + slot)
            outs.append({"wave": wave, "slot": slot, "ok": True})
        except Exception as e:  # noqa: BLE001
            outs.append({"wave": wave, "slot": slot, "ok": False, "error": str(e)})

    for wave in range(cfg_wave["p2_waves"]):
        threads = []
        for slot in range(cfg_wave["p2_wave_size"]):
            t = threading.Thread(target=one, args=(wave, slot))
            t.start()
            threads.append(t)
            sleep(float(cfg_wave["wave_stagger_s"]))
        for t in threads:
            t.join()
    return outs


# ---------------------------------------------------------------------------
# shared helpers for the point runners
# ---------------------------------------------------------------------------
def _install_check(ctx: GateContext, res: PointResult, expect_mode: str,
                   expect_candidates: list[int], expect_force=None) -> bool:
    """Verify the controller actually installed (a failed install silently
    falls back to static K - a classic false-PASS source)."""
    if ctx.install_log_ok is not None:
        ok = ctx.install_log_ok
        res.add("install.logged", ok,
                f"install log pre-checked by boot manager (ok={ok})",
                PASS if ok else FAIL)
        return ok
    lines = [l for _, l in ctx.logs.since(ctx.boot_ts) if INSTALL_SNIPPET in l]
    if not lines:
        res.add("install.logged", False,
                f"no '{INSTALL_SNIPPET}' line since boot - the controller fell "
                "back to static K (check candidates vs launch_k, incl. tiers)")
        return False
    line = lines[0]
    ok = True
    if expect_mode:
        m = f"mode={expect_mode}" in line or f"mode='{expect_mode}'" in line
        res.add("install.mode", m, f"expect mode={expect_mode}; line: {line[:160]}")
        ok &= m
    if expect_force:
        m = f"force={expect_force}" in line or f"force='{expect_force}'" in line
        res.add("install.force", m, f"expect force={expect_force}; line: {line[:160]}")
        ok &= m
    if expect_candidates:
        want = "candidates=[" + ",".join(str(c) for c in sorted(expect_candidates)) + "]"
        m = want in line.replace(" ", "")
        res.add("install.candidates", m, f"expect {want}; line: {line[:160]}")
        ok &= m
    m = f"launch_k={ctx.launch_k}" in line
    res.add("install.launch_k", m, f"expect launch_k={ctx.launch_k}; line: {line[:160]}")
    ok &= m
    return ok


def _require_instrument(ctx: GateContext, res: PointResult, point: str) -> bool:
    snap = ctx.metrics.fetch()
    if not any(k.startswith("vllm:spec_decode_num_sampled_draft_tokens_per_pos")
               for k in snap):
        res.add(f"{point}.A0.instrument", False,
                f"new debug counter {C_SAMPLES_POS} absent from /metrics - "
                "stage-1b patch not installed on this image", MISSING_INSTRUMENT)
        return False

    # Round 7: the C2 emitters are REQUEST-DRIVEN (the sched emitter fires per
    # scheduled step, the worker emitter per propose with requested_k > 0), so
    # "lines since boot" is only a meaningful instrumentation test once at
    # least one request has run. The boot manager sends that instrument probe
    # (see run_gate._boot) immediately after readiness, before any point
    # runner; this re-grep is the bounded-race guard for the log write (the
    # same pattern S7 uses). A MISSING_INSTRUMENT here therefore means the
    # probe ran and still nothing emitted - a real emitter/env-gate failure,
    # not the fresh-boot artifact the 2026-10-14 round-7 gate run produced
    # (its P1/P2/P5 A0 checked a healthy, correctly-gated boot before any
    # request existed and false-FAILED all three points).
    have_evidence = bool(split_src(ctx.evidence(), "worker")
                         or split_src(ctx.evidence(), "sched"))
    for _ in range(3):
        if have_evidence:
            break
        ctx.sleep(2.0)
        have_evidence = bool(split_src(ctx.evidence(), "worker")
                             or split_src(ctx.evidence(), "sched"))
    if not have_evidence:
        res.add(f"{point}.A0.instrument", False,
                f"no {TAG} evidence lines since boot (the boot-manager "
                f"instrument probe ran) - {ENV_EVIDENCE}=1 emitter missing",
                MISSING_INSTRUMENT)
        return False
    res.add(f"{point}.A0.instrument", True,
            "new debug counter and AWDBG1B evidence lines present")
    return True


def _window_series(ctx: GateContext, t0: float, t1: float | None = None) -> dict:
    recs = ctx.evidence(since=t0, until=t1)
    sched, worker = [], []
    for r in sorted(split_src(recs, "sched"),
                    key=lambda r: (r.get("seq") is None, r.get("seq") or 0)):
        sched.append({**r, "reqmap": {rq.get("id"): rq for rq in (r.get("reqs") or [])}})
    worker = sorted(split_src(recs, "worker"),
                    key=lambda r: (r.get("seq") is None, r.get("seq") or 0))
    return {"sched": sched, "worker": worker}


def _lag_check(sched: list[dict], res: PointResult, cid: str) -> None:
    """win(t) == requested_k(t-1) for requests in flight (verified) at both
    steps: the pending verify window keeps the PREVIOUS width while the next
    proposal changes.

    Trim exemption (round-2 attack: the phase model declares budget trims
    legitimate and P3.S4 deliberately provokes them): a row with
    win < requested_k(t-1) is a legitimate shortened window and is EXEMPT,
    not an offender, a trim never schedules more than was pending. The hard
    invariant, violated by neither trims nor the scheduler, is
    win > requested_k(t-1): a window LONGER than the previous proposal. The
    lag defect (window follows the CURRENT width) still fails this check
    through its grow-step rows (win(t) == requested_k(t) > requested_k(t-1)
    whenever the width rises); on shrink steps its rows are indistinguishable
    from trims and are exempt with them."""
    lag_total = lag_ok = trims = 0
    offenders = []
    for prev, cur in zip(sched, sched[1:]):
        rk_prev = prev.get("requested_k")
        if rk_prev is None:
            continue
        for rid, rq in cur["reqmap"].items():
            win = rq.get("win") or 0
            prev_rq = prev["reqmap"].get(rid)
            if not win or prev_rq is None or not (prev_rq.get("win") or 0):
                continue  # not verified at both steps
            lag_total += 1
            if win == rk_prev:
                lag_ok += 1
            elif win < rk_prev:
                trims += 1  # budget trim: legitimate (README phase model / P3.S4)
            elif len(offenders) < 5:
                offenders.append({"t": cur.get("t"), "id": rid, "win": win,
                                  "requested_k_prev": rk_prev})
    res.add(cid, lag_total > 0 and not offenders,
            f"in-flight (t-1,t) pairs {lag_total}, win(t)==requested_k(t-1) {lag_ok}, "
            f"legitimate trims (win < requested_k(t-1)) exempted: {trims}"
            + (f"; offenders (win > requested_k(t-1)): {offenders}"
               if offenders else ""))


def _row_account(s: dict | None, w: dict) -> tuple[list[str], int]:
    """Row accounting for one paired (sched, worker) line.

    C2 contract: the worker line lists EVERY batch row, and on a healthy
    boot the batch at propose time IS the scheduled set, so sched bs ==
    worker rows. Round 6 exemption (token-budget trims): a running request
    the token budget could not schedule this step is absent from
    num_scheduled_tokens (scheduled_running_reqs only appends actually
    scheduled requests, scheduler.py:828) while its row can persist in the
    worker's batch; such rows are engine-legitimate PROVIDED every row the
    scheduler DID schedule is still present exactly once and the surplus
    rows carry no scheduled tokens (their ids are absent from the sched
    line's reqs -- an unscheduled row has no verify window this step, so
    win == 0 by absence). Anything else stays an offender: a dropped or
    duplicated SCHEDULED row, or a surplus row claiming a scheduled id.
    Returns (problems, exempted_surplus_rows). A malformed reqs payload
    reports nothing here (never iterated unsafely; the alignment check
    reports the shape defect precisely)."""
    if s is None:
        return [], 0
    reqs = w.get("reqs")
    if not isinstance(reqs, list):
        return [], 0
    bs = s.get("bs") or 0
    if bs == len(reqs):
        return [], 0
    sched_ids = {rq.get("id") for rq in (s.get("reqs") or [])
                 if isinstance(rq, dict)}
    matched = [rq for rq in reqs
               if isinstance(rq, dict) and rq.get("id") in sched_ids]
    surplus = len(reqs) - len(matched)
    if (bs < len(reqs) and len(matched) == bs
            and len({rq.get("id") for rq in matched}) == len(matched)):
        return [], surplus
    return [f"sched bs={bs} != worker rows={len(reqs)} "
            f"(scheduled rows matched {len(matched)}, surplus {surplus}; "
            "C2: the worker line lists EVERY batch row and every scheduled "
            "row must appear exactly once; surplus rows are exempt only "
            "when the token budget left them unscheduled - id absent from "
            "the sched line's reqs)"], 0


def _unscheduled_row_count(s: dict | None, w: dict) -> int:
    """Worker rows whose id the paired sched line did NOT schedule this step
    (round 6): a row absent from num_scheduled_tokens carries no verify
    window (win == 0 by absence) and no scheduled tokens, yet it persists in
    the worker batch and samples like every row (worker truth), so A3c's
    executed-sample identity must cover bs + these rows. Malformed reqs
    payloads count as zero here (never iterated unsafely; the alignment
    check reports the malformation precisely)."""
    if s is None:
        return 0
    reqs = w.get("reqs")
    if not isinstance(reqs, list):
        return 0
    sched_ids = {rq.get("id") for rq in (s.get("reqs") or [])
                 if isinstance(rq, dict)}
    return sum(1 for rq in reqs
               if isinstance(rq, dict) and rq.get("id") not in sched_ids)


def _alignment_check(worker: list[dict], res: PointResult, cid: str,
                     sched: list[dict] | None = None) -> bool:
    """D-009 proposal-logit cache alignment, deterministic part (round 2).

    The proposal-logit cache is addressed per (req_state row, draft step):
    draft_logits is [max_num_reqs, num_speculative_steps, V] and the kernels
    index it as req_state_idx * stride_0 + draft_step * stride_1 (rej-utils
    150-300/330-810), with req_state_idx from the per-token expanded
    idx_mapping (V2 runner 1172-1288, propose store at :1907
    req_states.draft_tokens[input_batch.idx_mapping]) and the
    draft_step_idx >= num_speculative_steps rows skipped as bonus tokens.
    The stale-q' defect class D-009 proved greedy-invisible lives exactly
    here: a request reading a row/column another step or request wrote.

    The worker knows req_state_idx and draft_step_idx at write time, so the
    C2 worker line carries them (contract: reqs[].row, reqs[].steps) and this
    check asserts the write-side alignment deterministically:
      - row ownership: every request reports a row; rows are unique within a
        step (two requests must never share a req_state row);
      - per-token step alignment: a request's steps are exactly
        range(valid_len), contiguous from column 0, no skipped/reused step
        column, every step < k_eff; rows without draft positions (chunked
        prefill) have valid_len == 0 and steps == [].
    What it CANNOT see is the buffer VALUES (stale q' content): that remains
    P4.D4's distribution gate. An emitter without the fields predates the
    contract and is MISSING_INSTRUMENT, never a silent pass.

    Round-3 hardening (N5/N6):
      - the reqs payload itself is required (an emitter that omits it
        predates the contract -> MISSING_INSTRUMENT) and type-checked
        WITHOUT being iterated: a reqs value (or a reqs[] row, or a
        reqs[].steps value) of the wrong JSON type is reported as
        MISSING_MALFORMED_INSTRUMENT instead of crashing the gate run with
        a TypeError/AttributeError;
      - when the paired sched lines are supplied (P1 passes them; P2's B4
        owns row accounting there), the worker row count is cross-checked
        against sched bs, row accounting exactly like P2.B4, so an
        emitter that drops the reqs payload (or emits reqs: [] under a
        non-empty batch) fails loudly instead of passing vacuously over
        zero rows.

    Returns True when the fields were present and well-formed (the caller
    can gate on it).
    """
    # ---- structural pass: container/row/step-field types, checked WITHOUT
    # iterating any malformed value ----
    absent_reqs, malformed, missing_fields = [], [], []
    for w in worker:
        problem = _reqs_field_problem(w)
        if problem == "absent":
            absent_reqs.append(w.get("seq"))
            continue
        if problem is not None:
            malformed.append({"seq": w.get("seq"), "why": problem})
            continue
        for rq in w["reqs"]:
            if "row" not in rq or "steps" not in rq:
                missing_fields.append(w.get("seq"))
                break
            if not isinstance(rq["steps"], list):
                malformed.append({
                    "seq": w.get("seq"), "id": rq.get("id"),
                    "why": f"reqs[].steps is {type(rq['steps']).__name__} "
                           f"{rq['steps']!r}, not a JSON array"})
    if absent_reqs:
        res.add(cid, False,
                f"worker lines {absent_reqs[:5]} carry no reqs at all - the "
                "C2 contract requires the worker line to list EVERY batch "
                "row (reqs[].row + per-token reqs[].steps); an emitter "
                "without them predates the alignment contract - re-build the "
                "stage-1b patch", MISSING_INSTRUMENT)
        return False
    if malformed:
        res.add(cid, False,
                f"malformed instrumentation on worker lines "
                f"{[(m['seq'], m.get('why')) for m in malformed[:3]]} - reqs "
                "must be a JSON array of JSON objects with array-valued "
                "steps (C2 contract); failing the check instead of crashing "
                "the gate run", MISSING_MALFORMED_INSTRUMENT)
        return False
    if missing_fields:
        res.add(cid, False,
                f"worker lines {missing_fields[:5]} lack reqs[].row/steps - "
                "emitter predates the alignment contract (C2 requires row + "
                "per-token draft_step_idx); re-build the stage-1b patch",
                MISSING_INSTRUMENT)
        return False
    # ---- row accounting vs the paired sched bs (round-3 N6; P2's B4 owns
    # the same cross-check there, so P1 passes sched and P2 does not).
    # Round 6: token-budget trim surplus rows (ids absent from the paired
    # sched line's reqs) are exempt -- see _row_account for the exact
    # contract; a dropped or duplicated SCHEDULED row still fails. ----
    row_account = []
    trim_exempt_rows = 0
    if sched is not None:
        pairs, _us, _uw = join_by_seq(sched, worker)
        for s, w in pairs:
            problems, n_exempt = _row_account(s, w)
            trim_exempt_rows += n_exempt
            row_account.extend(
                {"seq": s.get("seq"), "why": p} for p in problems)
    # ---- content pass: the shape is now trusted, no access can crash ----
    bad_rows, bad_steps = [], []
    max_row = -1
    for w in worker:
        k_eff = w.get("k_eff")
        rows = [rq.get("row") for rq in w["reqs"]]
        if any(not isinstance(r, int) or r < 0 for r in rows):
            bad_rows.append({"seq": w.get("seq"), "why": "row missing/negative",
                             "rows": rows})
        elif len(rows) != len(set(rows)):
            bad_rows.append({"seq": w.get("seq"),
                             "why": "row collision (two requests share a "
                                    "req_state row - stale mapping or remap bug)",
                             "rows": rows})
        else:
            max_row = max([max_row] + [r for r in rows if isinstance(r, int)])
        for rq in w["reqs"]:
            vl, steps = rq.get("valid_len"), rq.get("steps") or []
            if k_eff is None:
                continue
            if vl not in (0, k_eff):
                bad_steps.append({"seq": w.get("seq"), "id": rq.get("id"),
                                  "why": f"valid_len {vl} not in (0, k_eff={k_eff})"})
            elif list(steps) != list(range(vl or 0)):
                bad_steps.append({"seq": w.get("seq"), "id": rq.get("id"),
                                  "why": f"steps {list(steps)[:6]} != "
                                         f"range(valid_len={vl}) (skipped or "
                                         "reused draft-step column)"})
            elif any(not isinstance(s, int) or s >= (k_eff or 0) or s < 0
                     for s in steps):
                bad_steps.append({"seq": w.get("seq"), "id": rq.get("id"),
                                  "why": f"step index out of [0, k_eff={k_eff}) "
                                         f"in {list(steps)[:6]}"})
    ok = not bad_rows and not bad_steps and not row_account
    res.add(cid, ok,
            f"row ownership unique+present (max row {max_row}), per-request "
            f"steps == range(valid_len), steps < k_eff"
            + (f"; row accounting: {row_account[:3]}" if row_account else "")
            + (f"; token-budget trim surplus rows exempt: {trim_exempt_rows}"
               if trim_exempt_rows else "")
            + (f"; row defects: {bad_rows[:3]}" if bad_rows else "")
            + (f"; step defects: {bad_steps[:3]}" if bad_steps else ""))
    return ok


# ---------------------------------------------------------------------------
# P1 - direct-work evidence (forced-K boot)
# ---------------------------------------------------------------------------
def run_p1(ctx: GateContext, res: PointResult) -> PointResult:
    launch_k = ctx.launch_k
    force_k = int(ctx.cfg["boot"]["force_k"])
    positions = list(range(launch_k))
    _assert_boot_phase(ctx, "P1")

    if not _require_instrument(ctx, res, "P1"):
        return res
    _install_check(ctx, res, "force", ctx.cfg["boot"]["candidates"], str(force_k))

    ctx.pair_mark()
    t0w = ctx.now()
    outs = run_steady(ctx.server, int(ctx.cfg["workload"]["p1_requests"]),
                      int(ctx.cfg["workload"]["p1_max_tokens"]), sleep=ctx.sleep)
    t1w = ctx.now()
    res.data["workload"] = outs
    snap1 = ctx.metrics.fetch()
    prev = ctx.pair_prev()

    drafts = delta(prev, snap1, C_DRAFTS)
    dtoks = delta(prev, snap1, C_DRAFT_TOKENS)
    samp = pos_delta(prev, snap1, C_SAMPLES_POS, positions)
    accp = pos_delta(prev, snap1, C_ACCEPTED_POS, positions)
    res.data["deltas"] = {"drafts": drafts, "draft_tokens": dtoks,
                          "sampled_per_pos": samp, "accepted_per_pos": accp}

    # A1.coverage (stage-1b sign-off, codex astra): the WINDOW itself must
    # contain evidence. A0.instrument only proves lines since boot; an emitter
    # stalled across exactly the P1 window would otherwise let A2 pass
    # vacuously on an empty series while vllm counters still advance.
    series = _window_series(ctx, t0w, t1w)
    empty_window = (len(series["sched"]) == 0 or len(series["worker"]) == 0)
    res.add("P1.A1.coverage", not empty_window,
            f"window evidence: {len(series['sched'])} sched / "
            f"{len(series['worker'])} worker lines",
            MISSING_INSTRUMENT if empty_window else None)
    if empty_window:
        return res

    if drafts <= 0:
        res.add("P1.A2.requested_k", False,
                "no spec-decode activity observed in the P1 window (drafts delta 0)")
        return res

    # A2 requested_k from evidence lines inside the scrape window
    sched = [s for s in series["sched"] if s.get("requested_k") is not None]
    bad = [s for s in sched if s.get("requested_k") != force_k]
    res.add("P1.A2.requested_k", not bad,
            f"{len(sched)} sched steps in window, {len(bad)} with "
            f"requested_k != {force_k}"
            + (f"; first offender: {json.dumps(bad[0])[:200]}" if bad else ""))
    res.add("P1.A2b.launch_k", all(s.get("launch_k") == launch_k for s in sched),
            f"launch_k == {launch_k} on all sched lines")

    # sched lines by seq, for per-line cross-checks that need the paired bs
    sched_by_seq = {s.get("seq"): s for s in series["sched"]}

    # A3 sampled positions: THE D-009 DISCRIMINATOR. Scheduler counters cannot
    # distinguish 'generated short' from 'generated wide then trimmed'
    # (sched-830-860 permits both); the drafter-side per-position sample
    # counts can.
    support_seen = {p for p in positions if samp[p] > 0}
    res.add("P1.A3.sampled_support", support_seen == set(range(force_k)),
            f"sampled-per-pos support {sorted(support_seen)}, want "
            f"{list(range(force_k))}; deltas {samp}")
    res.add("P1.A3b.no_sample_at_K", samp.get(force_k, 0) == 0,
            f"delta(samples at position {force_k}) == {samp.get(force_k, 0)}, want 0")
    total_samples = sum(samp[p] for p in range(force_k))
    k_eff_obs = (max(support_seen) + 1) if support_seen else 0
    # Round 4 (worker truth): the ar drafter samples EVERY batch row at
    # every position 0..k_eff-1, so the executed-sample counter must satisfy
    # sum(sampled_per_pos) == bs x k_eff on every worker line, with bs from
    # the PAIRED sched line (an independent source). Round 6: rows the token
    # budget left unscheduled persist in the worker batch and sample too, so
    # the identity's row count is bs + those rows (_unscheduled_row_count).
    # The old
    # `sum(samples) == force_k x delta(num_drafts)` form cannot hold on a
    # real boot and was retired: rows sample on their prefill/first-decode
    # steps BEFORE their first verify window exists (worker truth), so
    # samples and verify events diverge at window edges by design -- the
    # samples-vs-counter reconcile below covers the counter side instead.
    # The reqs payload is NEVER iterated here (the alignment check reports
    # a malformed one precisely, round-3 N5); the spp field is type-guarded
    # so a hostile payload fails its check instead of crashing the run.
    uni_bad = []
    line_spp_total = 0
    max_bs = 1
    trim_rows_total = 0
    for w in series["worker"]:
        k_eff = w.get("k_eff")
        if k_eff is None:
            continue
        # type-guard the spp field WITHOUT trusting it (round-3 principle:
        # a malformed payload fails its check, never the run)
        spp = w.get("sampled_per_pos")
        if not isinstance(spp, list):
            spp = []
        spp_sum = sum(v for v in spp
                      if isinstance(v, (int, float)) and not isinstance(v, bool))
        line_spp_total += spp_sum
        s = sched_by_seq.get(w.get("seq"))
        bs = (s.get("bs") or 0) if s is not None else None
        if bs is None:
            continue  # unpaired: A8b reports it; nothing to assert here
        max_bs = max(max_bs, bs)
        # Round 6: rows the token budget left unscheduled this step (ids
        # absent from the paired sched line's reqs) persist in the worker
        # batch and sample like every row (worker truth), so the identity
        # they must satisfy covers bs + those rows. Same exemption B4/A8's
        # row accounting got; a dropped SCHEDULED row shrinks the matched
        # count and still fails.
        n_unsched = _unscheduled_row_count(s, w)
        trim_rows_total += n_unsched
        if spp_sum != (bs + n_unsched) * k_eff:
            uni_bad.append({"seq": w.get("seq"), "bs": bs, "k_eff": k_eff,
                            "sum_spp": spp_sum})
    # counter-vs-evidence reconcile: the C1 counter folds each step's
    # samples into the NEXT step's stats (the post-step DraftTokenIds
    # channel), so the scrape-window counter delta and the evidence-line
    # sum differ by at most boundary steps x force_k x max bs.
    allowed = int(ctx.cfg.get("window_skew_steps", 2)) * force_k * max(1, max_bs)
    skew = abs(total_samples - line_spp_total)
    ok = k_eff_obs == force_k and not uni_bad and skew <= allowed
    res.add("P1.A3c.k_eff_matches_request", ok,
            f"observed k_eff={k_eff_obs} (max support + 1), want {force_k}; "
            f"uniform executed sampling "
            f"(sum(spp) == (bs + unscheduled-row surplus) x k_eff; "
            f"{trim_rows_total} token-budget trim surplus rows covered, "
            f"round 6) offenders: {uni_bad[:3] if uni_bad else 0}; "
            f"counter delta sum(samples)={total_samples} vs evidence lines "
            f"sum={line_spp_total} (skew {skew} vs tolerance {allowed} = "
            f"window_skew_steps x force_k x max bs: "
            f"{'within' if skew <= allowed else 'EXCEEDS'}; the C1 fold lands "
            "one step after the samples)")

    # A4a counter/evidence reconciliation: draft_tokens == sum of the
    # per-request verify windows (each row is one request; no bs multiply).
    # Boundary tolerance (round-2): the counter scrape window and the evidence
    # log window can differ by boundary steps; each boundary step contributes
    # at most launch_k x bs tokens.
    exp_dtoks = sum((rq.get("win") or 0)
                    for s in series["sched"] for rq in s["reqmap"].values())
    max_bs = max([len(s["reqmap"]) for s in series["sched"]] or [1])
    allowed_dtoks = int(ctx.cfg.get("window_skew_steps", 2)) * launch_k * max(1, max_bs)
    diff_dtoks = abs(dtoks - exp_dtoks)
    res.add("P1.A4a.window_counter_reconcile", diff_dtoks <= allowed_dtoks,
            f"delta(draft_tokens)={dtoks} vs evidence sum(win rows)={exp_dtoks}"
            + ("" if diff_dtoks == 0 else
               f" (diff {diff_dtoks} within boundary-skew tolerance "
               f"{allowed_dtoks} = window_skew_steps x launch_k x max bs)"))

    # A4b window bounds (round 4): a window is never longer than the
    # previous proposal -- win <= max(requested_k(t-1), launch_k); smaller
    # wins (trims) allowed. No cold-start clause: there is no launch-K
    # cold-start seeding on the sync path.
    viol = _window_bound_violations(series["sched"], launch_k)
    res.add("P1.A4b.window_bounds", not viol,
            "all verify windows within bounds (a window is never longer "
            "than the previous proposal: win <= max(requested_k(t-1), "
            "launch_k); smaller trims allowed; the launch_k term bounds "
            "only the window's first line)"
            + (f"; violations: {viol[:5]}" if viol else ""))

    # A5 no drafter work in the launch-only band
    res.add("P1.A5.no_launch_only_samples",
            all(samp[p] == 0 for p in range(force_k, launch_k)),
            f"sampled deltas at positions {list(range(force_k, launch_k))} are 0")

    # A6 returned valid lengths (worker truth). Round 4: rows WITHOUT draft
    # positions (valid_len == 0) are exempt exactly like A3c models them --
    # the offender is a row whose valid_len is neither 0 nor the forced
    # width. Lines with a malformed reqs field are skipped here (the
    # alignment check reports them - round-3 N5); a MISSING valid_len is the
    # alignment check's find, not this one's.
    workers = [w for w in series["worker"] if w.get("k_eff") is not None]
    zero_v = bad_v = 0
    first_bad = None
    for w in workers:
        if _reqs_field_problem(w) is not None:
            continue
        for rq in w["reqs"]:
            vl = rq.get("valid_len")
            if vl is None:
                continue
            if vl == 0:
                zero_v += 1
            elif vl != force_k:
                bad_v += 1
                if first_bad is None:
                    first_bad = (w.get("k_eff"), rq)
    res.add("P1.A6.valid_lengths", bad_v == 0,
            f"{len(workers)} worker steps; {bad_v} rows with valid_len not in "
            f"(0, {force_k}) ({zero_v} rows without draft positions exempt, "
            "like A3c)" + (f"; first: {first_bad}" if first_bad else ""))
    res.add("P1.A6b.k_eff", all(w.get("k_eff") == force_k for w in workers),
            f"k_eff == {force_k} on all worker lines (n={len(workers)})")

    # A7 accepted-per-position reconciliation in the launch-only band
    res.add("P1.A7.accepted_pos_reconcile",
            all(accp[p] == 0 for p in range(force_k, launch_k)),
            f"accepted_per_pos deltas {accp}: positions >= {force_k} must be 0")

    # A8 D-009 proposal-logit cache alignment (deterministic write-side part;
    # see _alignment_check for the contract and its sources). The sched lines
    # are supplied so the worker row count is cross-checked against sched bs
    # (row accounting, like P2.B4) - a dropped reqs payload fails loudly
    # instead of passing vacuously over zero rows (round-3 N6).
    _alignment_check(series["worker"], res, "P1.A8.alignment_rows",
                     sched=series["sched"])

    # A8b pairing (stage-1b sign-off, codex astra): a worker line whose seq
    # pairs with no sched line escapes row accounting entirely (join_by_seq
    # ignores it). Blocking under requirement 1: such a line's ownership
    # cannot be verified, so it must not ride along silently.
    sched_seqs = {s.get("seq") for s in series["sched"]}
    unmatched = [w for w in series["worker"] if w.get("seq") not in sched_seqs]
    res.add("P1.A8b.pairing", not unmatched,
            f"{len(unmatched)} worker lines pair with no sched line"
            + (f"; first offender seq={unmatched[0].get('seq')}" if unmatched else ""))

    # Round 8 (D-014 conditions 1+2): the runtime fingerprints and the C2
    # field completion. The FP checks gate the boot's PROOF surface (an
    # absent fingerprint layer is MISSING_INSTRUMENT); the C2 checks join
    # proposals with verify events and commits per request.
    _awfp_boot_check(ctx, res, "P1")
    _c2_semantics_check(series, res, "P1.C2", launch_k)
    return res


def _window_bound_violations(sched: list[dict], launch_k: int) -> list[dict]:
    """A window can never be LONGER than the previous proposal (round 4).

    win(t) == the previous proposal's width (k_eff(t-1), one-step lag) or a
    budget trim of it, so win(t) <= requested_k(t-1) on a healthy drafter;
    the launch_k term only bounds the evidence window's first line (its
    predecessor lies outside the window). The old cold-start clause
    (win == launch_k on a non-first verify = violation) encoded the
    pre-stage-1b carrier model and is GONE: with the sync path's
    proposal-width carrier there is no launch-K cold-start seeding, and a
    launch_k-width window on a non-first verify is exactly what the pinned
    drafter produces -- which this bound already flags on every shrink step
    (win > requested_k(t-1))."""
    viol = []
    for i, s in enumerate(sched):
        rk_prev = sched[i - 1].get("requested_k") if i else None
        for rid, rq in s["reqmap"].items():
            win = rq.get("win") or 0
            if not win:
                continue
            if win > max(rk_prev or 0, launch_k):
                viol.append({"t": s.get("t"), "id": rid, "win": win,
                             "requested_k_prev": rk_prev})
    return viol


# ---------------------------------------------------------------------------
# P2 - in-process transitions (armed oscillation boot)
# ---------------------------------------------------------------------------
def run_p2(ctx: GateContext, res: PointResult) -> PointResult:
    launch_k = ctx.launch_k
    cands = sorted(set(int(c) for c in ctx.cfg["boot"]["candidates"]))
    positions = list(range(launch_k))
    _assert_boot_phase(ctx, "P2")
    if not _require_instrument(ctx, res, "P2"):
        return res
    _install_check(ctx, res, "armed", cands, "oscillate")

    ctx.pair_mark()
    outs = run_waves(ctx.server, ctx.cfg["workload"], sleep=ctx.sleep)
    snap1 = ctx.metrics.fetch()
    prev = ctx.pair_prev()
    res.data["waves"] = outs
    res.data["deltas"] = {
        "drafts": delta(prev, snap1, C_DRAFTS),
        "draft_tokens": delta(prev, snap1, C_DRAFT_TOKENS),
        "sampled_per_pos": pos_delta(prev, snap1, C_SAMPLES_POS, positions),
    }
    series = _window_series(ctx, ctx.boot_ts)
    sched, worker = series["sched"], series["worker"]
    if len(sched) < 8 or len(worker) < 8:
        res.add("P2.coverage", False,
                f"insufficient evidence steps (sched={len(sched)}, worker={len(worker)})",
                MISSING_INSTRUMENT)
        return res

    # B1 square wave in requested_k over decode steps
    ks = [s.get("requested_k") for s in sched if s.get("requested_k") is not None]
    flips = sum(1 for a, b in zip(ks, ks[1:]) if a != b)
    cycles = flips / 2.0
    need = float(ctx.cfg["workload"].get("p2_cycles", 3))
    res.add("P2.B1.square_wave", cycles >= need,
            f"requested_k head={ks[:12]} flips={flips} cycles={cycles:.1f} "
            f"(need >= {need})")

    # B2 one-step lag (pending verify keeps the previous width)
    _lag_check(sched, res, "P2.B2.one_step_lag")

    # B3 worker samples follow the CURRENT width. Round 4: a WINDOWLESS step
    # (the paired sched line scheduled no draft positions on any row -- the
    # boot's first scheduled batch is one) is exempt from the support
    # assertion: there is no window coupling to assert there, and the
    # drafter's samples on such a step are worker truth (the ar drafter
    # samples every batch row whether or not a window is scheduled). The
    # support assertion still applies to every step with a scheduled
    # window, and k_eff must be a legal ladder candidate everywhere.
    sched_by_seq = {s.get("seq"): s for s in sched}
    w_ok = w_total = w_windowless = 0
    w_off = []
    for w in worker:
        k_eff = w.get("k_eff")
        spp = w.get("sampled_per_pos") or []
        if k_eff is None:
            continue
        w_total += 1
        support = {i for i, v in enumerate(spp) if v > 0}
        s = sched_by_seq.get(w.get("seq"))
        windowless = (
            s is not None
            and bool(s.get("reqmap"))
            and all((rq.get("win") or 0) == 0 for rq in s["reqmap"].values())
        )
        if support == set(range(k_eff)) and k_eff in cands:
            w_ok += 1
        elif windowless and k_eff > 0:
            w_ok += 1
            w_windowless += 1
        elif len(w_off) < 5:
            w_off.append({"seq": w.get("seq"), "k_eff": k_eff, "support": sorted(support)})
    res.add("P2.B3.samples_follow_current", w_total > 0 and w_ok == w_total,
            f"worker steps {w_total}, consistent {w_ok} (support == 0..k_eff-1, "
            f"k_eff in {cands}; {w_windowless} windowless steps exempt)"
            + (f"; offenders: {w_off}" if w_off else ""))

    # B4 aggregate identity per step (bs>1) + observed reordering. Round-2:
    # rows WITHOUT draft positions (chunked prefill) are excluded from the
    # identity, the identity is worker-side exact (sum(spp) == sum(valid_len)
    # over the same line), uniform-sampling is asserted per row
    # (valid_len in {0, k_eff}), and the sched bs must equal the worker row
    # count (row accounting; a worker line that drops rows breaks it loudly).
    # Round 6: token-budget trim surplus rows (ids absent from the paired
    # sched line's reqs) are exempt from that equality -- a row the budget
    # left unscheduled is absent from num_scheduled_tokens while it can
    # persist in the worker batch (win == 0 by absence); a dropped or
    # duplicated SCHEDULED row still fails (_row_account).
    pairs, us, uw = join_by_seq(sched, worker)
    res.add("P2.B4a.pairing", not (us or uw),
            f"sched/worker seq pairs {len(pairs)}; unpaired sched {len(us)} "
            f"worker {len(uw)}", PASS if not (us or uw) else WARN)
    ident_ok = ident_total = 0
    ident_bad = []
    prefill_rows_total = 0
    trim_exempt_rows = 0
    for s, w in pairs:
        k_eff = w.get("k_eff")
        reqs = w.get("reqs") or []
        if k_eff is None:
            continue
        ident_total += 1
        spp_sum = sum(w.get("sampled_per_pos") or [])
        # round-3 N5: a malformed reqs field (not a JSON array of objects)
        # is an offender here - NEVER iterated, so it cannot crash the run;
        # the alignment check reports the precise shape defect
        reqs_problem = _reqs_field_problem(w)
        if reqs_problem is not None:
            ident_bad.append({"seq": s.get("seq"),
                              "why": [f"reqs field {reqs_problem} "
                                      "(malformed instrumentation)"]})
            continue
        vl_sum = sum(rq.get("valid_len") or 0 for rq in reqs)
        n_prefill = sum(1 for rq in reqs if (rq.get("valid_len") or 0) == 0)
        prefill_rows_total += n_prefill
        problems = []
        if spp_sum != vl_sum:
            problems.append(f"sum(spp)={spp_sum} != sum(valid_len)={vl_sum}")
        if any((rq.get("valid_len") or 0) not in (0, k_eff) for rq in reqs):
            problems.append("valid_len not in {0, k_eff} on some row")
        # Round 6: row accounting with the token-budget trim exemption
        # (surplus rows whose id the sched line did not schedule are
        # engine-legitimate; a dropped or duplicated SCHEDULED row still
        # fails -- see _row_account).
        acct_problems, n_exempt = _row_account(s, w)
        trim_exempt_rows += n_exempt
        problems.extend(acct_problems)
        if problems:
            ident_bad.append({"seq": s.get("seq"), "why": problems})
        else:
            ident_ok += 1
    res.add("P2.B4.reorder_identity", ident_total > 0 and ident_ok == ident_total,
            f"step identities {ident_ok}/{ident_total} (sum(samples)=="
            f"sum(valid_len); rows without draft positions excluded: "
            f"{prefill_rows_total}; uniform valid_len in (0, k_eff); "
            f"bs == worker rows, token-budget trim surplus exempt: "
            f"{trim_exempt_rows} rows)"
            + (f"; offenders: {ident_bad[:3]}" if ident_bad else ""))

    # B4b reorder observation, emitter-independent (round-2 attack: the C2
    # contract does not pin emitter ordering, so a canonical-order emitter
    # would leave reordering permanently unobserved). Two channels:
    # (a) order-compare, adjacent ordered id tuples differ (needs the pinned
    #     batch-order emitter, see README C2);
    # (b) set-churn compare, adjacent id SETS differ (works for any emitter).
    # PASS if either fires; WARN only when neither did.
    order_events = set_events = 0
    prev_order = prev_set = None
    for s in sched:
        order = tuple(rid for rid, rq in s["reqmap"].items() if (rq.get("win") or 0) > 0)
        cur_set = frozenset(order)
        if prev_order is not None and order != prev_order:
            order_events += 1
        if prev_set is not None and cur_set != prev_set:
            set_events += 1
        prev_order, prev_set = order, cur_set
    if order_events >= 1:
        b4b_status, b4b_why = PASS, ""
    elif set_events >= 1:
        b4b_status, b4b_why = PASS, (" (observed via set churn; order-compare "
                                     "needs the batch-order emitter contract)")
    else:
        b4b_status, b4b_why = WARN, " (the workload never reordered)"
    res.add("P2.B4b.reorder_observed", b4b_status != WARN,
            f"running-set changes observed: order-compare {order_events}, "
            f"set-compare {set_events} (bs>1 with staggered lifetimes must "
            f"reorder){b4b_why}", b4b_status)

    # B4c emitter-order contract (round-2 attack): if every multi-row window
    # is EXACTLY id-sorted while bs>1, the emitter is almost certainly
    # canonicalizing instead of preserving batch order, order-compare is then
    # blind and the C2 contract is violated. FAIL-class, never silent.
    multi = [tuple(rid for rid, rq in s["reqmap"].items() if (rq.get("win") or 0) > 0)
             for s in sched]
    multi = [t for t in multi if len(t) > 1]
    canonical = multi and all(t == tuple(sorted(t)) for t in multi)
    res.add("P2.B4c.emitter_order_contract", not canonical,
            ("sched reqs order is id-sorted on EVERY multi-row step while "
             "bs>1 - the emitter appears to canonicalize instead of "
             "preserving scheduler batch order; B4b's order-compare is blind "
             "(C2 contract: emit batch order, see README)")
            if canonical else
            f"batch-order emission plausible ({len(multi)} multi-row steps, "
            "not all id-sorted)",
            PASS if not canonical else MISSING_INSTRUMENT)

    # B6 D-009 proposal-logit cache alignment during oscillation: per-step
    # row ownership + per-request step columns, across reordering ...
    _alignment_check(worker, res, "P2.B6.alignment_rows")
    # ... and specifically ACROSS WIDTH TRANSITIONS: when k_eff changes
    # between adjacent paired steps, every request sampling in the later step
    # must FULLY re-cover 0..k_eff-1 (no stale step column reused across the
    # transition, the grow-back step is where a stale column would hide).
    trans_total = trans_ok = 0
    trans_bad = []
    for (s1, w1), (s2, w2) in zip(pairs, pairs[1:]):
        k1, k2 = w1.get("k_eff"), w2.get("k_eff")
        if k1 is None or k2 is None or k1 == k2:
            continue
        trans_total += 1
        bad = []
        # round-3 N5: never iterate a malformed reqs field or a non-array
        # steps value - the alignment check (B6, which ran above) reports the
        # shape defect; here it only must not crash the run
        if _reqs_field_problem(w2) is not None:
            continue
        for rq in w2["reqs"]:
            vl = rq.get("valid_len") or 0
            steps = rq.get("steps")
            if steps is not None and not isinstance(steps, list):
                continue  # shape defect already reported by B6
            if vl and list(steps or []) != list(range(vl)):
                bad.append({"id": rq.get("id"), "valid_len": vl,
                            "steps": list((steps or []))[:6]})
        if bad:
            trans_bad.append({"seq": s2.get("seq"), "k_eff": [k1, k2],
                              "rows": bad})
        else:
            trans_ok += 1
    if trans_total:
        res.add("P2.B6b.transition_alignment", trans_ok == trans_total,
                f"width transitions checked {trans_total}, full re-cover "
                f"(steps == range(valid_len) after every k_eff change) {trans_ok}"
                + (f"; offenders: {trans_bad[:3]}" if trans_bad else ""))
    else:
        # no k_eff change in the window: nothing to assert across transitions.
        # SKIP (not FAIL) keeps the documented division of labor - a pinned
        # drafter (k_eff constant while the scheduler oscillates) is P5.E2's
        # defect, and B1 already fails when the square wave is absent.
        res.add("P2.B6b.transition_alignment", True,
                "no k_eff transition in the window - nothing to assert across "
                "transitions (oscillation presence is B1's check; a pinned "
                "drafter is P5.E2's defect and also fails B2's lag bound)",
                SKIP)

    # B6c row remapping across reordering (coverage): requests MAY move
    # req_state rows between steps (idx_mapping is re-derived per step); the
    # defect the alignment checks catch is sharing/skipping, not moving. WARN
    # if the workload never remapped (weaker reordering coverage).
    remaps = 0
    row_by_id: dict = {}
    for s, w in pairs:
        if _reqs_field_problem(w) is not None:
            continue  # shape defect reported by B6 - never crash here (round-3 N5)
        for rq in w["reqs"]:
            rid, row = rq.get("id"), rq.get("row")
            if rid is None or not isinstance(row, int):
                continue
            if rid in row_by_id and row_by_id[rid] != row:
                remaps += 1
            row_by_id[rid] = row
    res.add("P2.B6c.row_remap_observed", remaps >= 1,
            f"request row remaps observed across steps: {remaps} "
            "(idx_mapping is re-derived per step; moving is legitimate, "
            "sharing is not)",
            PASS if remaps >= 1 else WARN)

    # B5 no stale suffix after shrink
    kmin = cands[0]
    stale = 0
    for w in worker:
        spp = w.get("sampled_per_pos") or []
        if w.get("k_eff") == kmin:
            stale += sum(1 for p in range(kmin, launch_k)
                         if p < len(spp) and spp[p] > 0)
    res.add("P2.B5.no_stale_suffix", stale == 0,
            f"sample events at positions >= {kmin} while k_eff == {kmin}: {stale} "
            "(shrink must invalidate the suffix; the launch-only band stays silent)")

    # Round 8 (D-014 conditions 1+2): the runtime fingerprints and the C2
    # field completion during oscillation (the join spans width transitions;
    # the argmax evidence covers the greedy rows of every width).
    _awfp_boot_check(ctx, res, "P2")
    _c2_semantics_check(series, res, "P2.C2", launch_k)
    return res


# ---------------------------------------------------------------------------
# P3 - boundary coverage
# ---------------------------------------------------------------------------
def run_p3(ctx: GateContext, res: PointResult) -> PointResult:
    launch_k = ctx.launch_k
    force_k = int(ctx.cfg["boot"]["force_k"])
    positions = list(range(launch_k))
    _assert_boot_phase(ctx, "P3")

    # S7 async exclusion (round 5 rewrite). The round-4 form grepped the
    # container ARGV only and reported "async flags: none" while the boot
    # actually RAN the AsyncScheduler: SchedulerConfig.async_scheduling=None
    # auto-resolves to True for MTP boots ("mtp" is an EagleModelTypes
    # member, the executor supports async, no incompatibility fired), the
    # engine core then never calls update_draft_token_ids, and the C1 chain
    # starved. Two layers now:
    #   1. the harness boots pass --no-async-scheduling (env EXTRA append in
    #      prepare_config_dir), so the documented sync path really runs;
    #   2. THIS check verifies ENGINE TRUTH: the patched scheduler's
    #      update_draft_token_ids logs SYNC_CHANNEL_SNIPPET once on its first
    #      call -- impossible under the async scheduler. A tiny probe request
    #      guarantees the channel has run by now (a P3-only run would
    #      otherwise assert before any engine step).
    ctx.server.chat([{"role": "user", "content": "Reply with OK."}],
                    max_tokens=4, temperature=0.0)
    # Bounded re-grep: the engine core queues the step's outputs BEFORE
    # post_step logs the line, so on a P3-only run against a fresh boot the
    # first grep can race the log write (the default P1-then-P3 order has
    # minutes of margin).
    sync_lines: list[str] = []
    for _ in range(3):
        sync_lines = [l for _, l in ctx.logs.since(ctx.boot_ts)
                      if SYNC_CHANNEL_SNIPPET in l]
        if sync_lines:
            break
        ctx.sleep(2.0)
    async_flags = argv_async_flags(ctx.argv)
    res.add("P3.S7.async_excluded", bool(sync_lines) and not async_flags,
            f"sync draft-token channel live lines: {len(sync_lines)} "
            f"(engine truth: update_draft_token_ids ran -> the sync scheduler "
            "is serving; the line cannot appear under the async scheduler, "
            "which the round-4 run proved by silently auto-enabling it for "
            f"MTP); container argv async-scheduling ENABLE flags: "
            f"{async_flags or 'none'} "
            "(secondary signal only -- argv cannot expose the auto-enable; "
            f"the harness's own {NO_ASYNC_FLAG} boot flag is a negation and "
            "is not counted, round 6)")

    if not _require_instrument(ctx, res, "P3"):
        return res
    _install_check(ctx, res, "force", ctx.cfg["boot"]["candidates"], str(force_k))

    def _first_win_of_new(since: float, until: float | None) -> tuple[int | None, str]:
        """First verify window of requests first seen in [since, until]."""
        pre_ids = {rq.get("id")
                   for s in _window_series(ctx, ctx.boot_ts, since)["sched"]
                   for rq in s["reqmap"].values()}
        win = None
        who = ""
        for s in _window_series(ctx, since, until)["sched"]:
            for rid, rq in s["reqmap"].items():
                if rq.get("win") and rid not in pre_ids:
                    return rq["win"], str(rid)
                if rq.get("win") and win is None:
                    win, who = rq["win"], str(rid)
        return win, who

    # S1 first proposal: quiesce, then one fresh request
    ctx.sleep(float(ctx.cfg.get("quiesce_s", 3.0)))
    t_s1 = ctx.now()
    ctx.server.chat([{"role": "user", "content": "List the first 12 Fibonacci numbers."}],
                    max_tokens=48, temperature=0.0)
    t_s1e = ctx.now()
    win1, who1 = _first_win_of_new(t_s1, t_s1e)
    # Round 4 (k_eff semantics): the sync path has NO launch-K cold-start
    # seeding -- request.spec_token_ids is only ever set from the
    # DraftTokenIds output -- so a request's FIRST verify window equals its
    # first proposal's own width (one-step lag). On this boot that is the
    # forced width. Asserting win == launch_k here would pass on the
    # pre-stage-1b defective engine (launch-width carrier) and fail the
    # correct one -- inverted discrimination.
    res.add("P3.S1.first_proposal_coldstart", win1 == force_k,
            f"first verify window of a request first seen in the window: "
            f"{win1} (req {who1}), expect {force_k} (the request's first "
            "verify keeps its first proposal's OWN width -- one-step lag, "
            "k_eff semantics; there is no launch-K cold-start seeding on "
            "the sync path, so win == launch_k here would be the pre-1b "
            "launch-width-carrier defect)")
    ctx.pair_mark()
    ctx.server.chat([{"role": "user", "content": "Count 5 to 15."}],
                    max_tokens=32, temperature=0.0)
    snap_a = ctx.metrics.fetch()
    samp_a = pos_delta(ctx.pair_prev(), snap_a, C_SAMPLES_POS, positions)
    res.add("P3.S1.no_launch_only_samples",
            all(v == 0 for p, v in samp_a.items() if p >= force_k),
            f"sampled deltas after the first proposal {samp_a}: "
            f"positions >= {force_k} silent")

    # S2 chunked prefill: prompt above max_num_batched_tokens (8192)
    long_prompt = long_ctx(12000, "S2CH") + "\n\nSummarize the log in two sentences."
    t_s2 = ctx.now()
    r2 = ctx.server.chat([{"role": "user", "content": long_prompt}],
                         max_tokens=48, temperature=0.0)
    t_s2e = ctx.now()
    win2, _ = _first_win_of_new(t_s2, t_s2e)
    res.add("P3.S2.chunked_first_window", win2 == force_k,
            f"chunked-prefill request first verify window {win2}, expect "
            f"{force_k} (the chunked request's first verify keeps its first "
            "kept proposal's OWN width -- one-step lag, k_eff semantics, "
            "same round-4 model as S1; the pre-1b engine would show "
            "launch_k here)")
    res.add("P3.S2.chunked_completed", bool(r2), "long request completed")

    # S3 mixed batch: long-prefill request in flight while decodes run.
    # Round-5 shape change (the 2026-10-14 run saw 0 mixed steps): one short
    # 32-token request fired 1s into a 9000-token prefill never overlapped a
    # verifying row -- the 8192-token chunk consumed the whole token budget
    # and the short request finished before the long row's first verify. The
    # new shape fires several LONGER decodes once the long request is
    # PAST its chunks (staggered), so a joining request's prefill step (win 0)
    # reliably shares a step with a verifying row (win > 0). The check itself
    # is unchanged and stays FAIL-strict; if the trigger still never fires,
    # that is a reported coverage gap, not a relaxed threshold.
    box = {}
    p3cfg = ctx.cfg.get("p3", {}) or {}
    s3_shorts = int(p3cfg.get("s3_shorts", 3))
    s3_short_max = int(p3cfg.get("s3_short_max_tokens", 192))
    s3_stagger = float(p3cfg.get("s3_stagger_s", 1.5))

    def _long_mixed():
        try:
            box["long"] = ctx.server.chat(
                [{"role": "user", "content": long_ctx(9000, "S3MX") + "\n\nSummarize."}],
                max_tokens=64, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            box["long"] = {"error": str(e)}

    t_s3 = ctx.now()
    th = threading.Thread(target=_long_mixed)
    th.start()
    ctx.sleep(1.0)
    short_threads = []
    for i in range(s3_shorts):
        st = threading.Thread(target=lambda i=i: ctx.server.chat(
            [{"role": "user",
              "content": f"Mixed-batch probe {i}: name {8 + i} common birds."}],
            max_tokens=s3_short_max, temperature=0.0, seed=300 + i))
        st.start()
        short_threads.append(st)
        ctx.sleep(s3_stagger)
    th.join()
    for st in short_threads:
        st.join()
    t_s3e = ctx.now()
    sched3 = _window_series(ctx, t_s3, t_s3e)["sched"]
    mixed_steps = 0
    for s in sched3:
        wins = [rq.get("win") or 0 for rq in s["reqmap"].values()]
        if len(wins) >= 2 and any(w > 0 for w in wins) and any(w == 0 for w in wins):
            mixed_steps += 1
    res.add("P3.S3.mixed_steps_seen", mixed_steps >= 1,
            f"steps mixing spec-decode and non-spec requests: {mixed_steps} "
            f"(shape: 9000-token prefill + {s3_shorts} decodes x "
            f"{s3_short_max} tokens staggered {s3_stagger}s)")

    # S4 shortened windows: budget pressure from concurrency + a calibrated
    # chunk-boundary prompt. Round-7 retune (both 2026-10-14 runs saw zero
    # trims): a trim fires only on a step whose token-budget leftover after
    # the running loop is comparable to the decode rows' pending-window
    # demand, so the shape (a) raises concurrency - seven long-lived shorts,
    # all decoding with pending windows - and (b) calibrates the long prompt
    # at runtime onto small budget-leftover targets (see _s4_shape; the
    # tokenized length is measured from the server's own usage, and each
    # attempt rotates its first-sentence salt digit so the prefix cache
    # cannot absorb the chunk). Attempts stop at the first one that trims.
    # The scheduler source (running loop: num_new_tokens = min(win,
    # token_budget, input_budget - draft_slots); FCFS promotion) suggests the
    # trigger may remain structurally unreachable - the check detail and
    # res.data report the calibrated shape so the next live run settles it.
    p3cfg = ctx.cfg.get("p3", {}) or {}
    s4_shape = _s4_shape(ctx, p3cfg)
    res.data["s4_shape"] = s4_shape["notes"]
    t_s4 = ctx.now()
    shortened: list[dict] = []
    sched4: list[dict] = []
    for attempt_i, prompt in enumerate(s4_shape["attempts"]):
        t_a = ctx.now()
        threads = []
        n_shorts = (int(p3cfg.get("s4_shorts", 7)) if attempt_i == 0
                    else int(p3cfg.get("s4_topup", 3)))
        for i in range(n_shorts):
            th = threading.Thread(target=lambda i=i, a=attempt_i: ctx.server.chat(
                [{"role": "user",
                  "content": f"Explain topic {a}-{i}: brief note."}],
                max_tokens=int(p3cfg.get("s4_short_max_tokens", 160)),
                temperature=0.0, seed=200 + a * 10 + i))
            th.start()
            threads.append(th)
            ctx.sleep(float(p3cfg.get("s4_stagger_s", 0.3)))
        threads.append(threading.Thread(
            target=lambda p=prompt: ctx.server.chat(
                [{"role": "user", "content": p}],
                max_tokens=32, temperature=0.0)))
        threads[-1].start()
        for t in threads:
            t.join()
        win_sched = _window_series(ctx, t_a, ctx.now())["sched"]
        sched4.extend(win_sched)
        for i, s in enumerate(win_sched):
            if i == 0 or s.get("requested_k") is None:
                continue
            prev_k = win_sched[i - 1].get("requested_k")
            for rid, rq in s["reqmap"].items():
                win = rq.get("win") or 0
                if win and prev_k is not None and win < prev_k:
                    shortened.append({"attempt": attempt_i, "t": s.get("t"),
                                      "id": rid, "win": win,
                                      "requested_k_prev": prev_k})
        if shortened:
            break
    res.data["shortened_examples"] = shortened[:5]
    if shortened:
        res.add("P3.S4.shortened_windows", True,
                f"trimmed windows observed: {len(shortened)} (e.g. {shortened[0]}); "
                "generate>verify divergence is legitimate here, the invariant is "
                "that a trimmed window never schedules more than the drafter "
                "produced")
        wmap = {w.get("seq"): w for w in _window_series(ctx, t_s4, ctx.now())["worker"]}
        bad_trim = []
        for s in sched4:
            w = wmap.get(s.get("seq"))
            k_eff = w.get("k_eff") if w else None
            for rid, rq in s["reqmap"].items():
                win = rq.get("win") or 0
                if k_eff is not None and 0 < win <= (s.get("requested_k") or 0) \
                        and win != launch_k and win > k_eff:
                    bad_trim.append({"seq": s.get("seq"), "id": rid,
                                     "win": win, "k_eff": k_eff})
        res.add("P3.S4b.trim_within_k_eff", not bad_trim,
                "every trimmed window satisfies win <= k_eff (the drafter "
                "produced at least what the scheduler scheduled)"
                + (f"; violations: {bad_trim[:5]}" if bad_trim else ""))
    else:
        res.add("P3.S4.shortened_windows", True,
                "no trimmed window observed under the retuned load "
                f"(shape: {json.dumps(s4_shape['notes'])[:220]}) - trigger "
                "needs tuning: more concurrency or tighter input budget "
                "(round 7 calibrated the prompt length from live usage; see "
                "README 'S4 trigger analysis')", WARN)

    # S5 rejection: low-acceptance workload
    ctx.pair_mark()
    ctx.server.chat([{"role": "user", "content": P4_PROMPTS[4][1]}],
                    max_tokens=96, temperature=1.0, seed=99)
    snap5 = ctx.metrics.fetch()
    prev5 = ctx.pair_prev()
    drafts5 = delta(prev5, snap5, C_DRAFTS)
    acc5 = delta(prev5, snap5, C_ACCEPTED)
    rate = (acc5 / drafts5) if drafts5 else None
    accp5 = pos_delta(prev5, snap5, C_ACCEPTED_POS, positions)
    res.add("P3.S5.rejection_observed",
            rate is not None and rate < 0.9 * max(1, force_k - 1),
            f"acceptance rate {rate} (accepted {acc5} / drafts {drafts5}); "
            f"per-pos deltas {accp5}")

    # S6 finish/reuse: identical long prompt twice (prefix reuse) + successor
    # cold start. Round-5 probe shape: the round-4 probe was an ~8-token
    # prompt -- far below one 1,600-token prefix-cache block -- so the second
    # request could never record a hit (the live run: hits delta 0.0). The
    # identical prefix now exceeds one block; the tokenized length is
    # verified from the server's own usage (prompt_tokens, and
    # prompt_tokens_details.cached_tokens when the API reports it -- the
    # boots run --enable-prompt-tokens-details), and a positive hit is
    # required from the counter OR the cached-tokens detail.
    p3cfg = ctx.cfg.get("p3", {}) or {}
    # long_ctx yields ~0.83 tokenized tokens per requested target token
    # (~38-token sentences at reps = target//60); 3000 lands ~1.9k tokenized
    # tokens, comfortably above the 1,600-token block threshold.
    s6_target = int(p3cfg.get("s6_prefix_target_tokens", 3000))
    s6_block = int(p3cfg.get("prefix_block_tokens", PREFIX_BLOCK_TOKENS))
    prompt = long_ctx(s6_target, "S6PX") + "\n\nReply with OK."
    r6_first = ctx.server.chat([{"role": "user", "content": prompt}],
                               max_tokens=24, temperature=0.0, seed=5)
    ctx.pair_mark()
    t_s6 = ctx.now()
    r6 = ctx.server.chat([{"role": "user", "content": prompt}], max_tokens=24,
                         temperature=0.0, seed=5)
    t_s6e = ctx.now()
    snap6 = ctx.metrics.fetch()
    hits6 = delta(ctx.pair_prev(), snap6, C_PREFIX_HITS)
    win6, _ = _first_win_of_new(t_s6, t_s6e)

    def _usage(resp: dict) -> tuple[int | None, int | None]:
        u = (resp or {}).get("usage") or {}
        ptoks = u.get("prompt_tokens")
        det = u.get("prompt_tokens_details") or {}
        cached = det.get("cached_tokens")
        return (int(ptoks) if isinstance(ptoks, (int, float)) else None,
                int(cached) if isinstance(cached, (int, float)) else None)

    ptoks1, cached1 = _usage(r6_first)
    ptoks2, cached2 = _usage(r6)
    length_ok = (ptoks1 is not None and ptoks1 >= s6_block) or \
                (ptoks2 is not None and ptoks2 >= s6_block)
    hit_ok = hits6 > 0 or (cached2 or 0) > 0
    res.add("P3.S6.prefix_reuse", length_ok and hit_ok,
            f"probe prompt_tokens={ptoks1}/{ptoks2} (block threshold "
            f"{s6_block}); second-request prefix-cache hits delta {hits6}, "
            f"cached_tokens={cached2} (first request cached {cached1}); "
            "the probe must exceed one prefix-cache block (the round-4 "
            "8-token probe could never hit) and the identical second request "
            "must record a positive hit")
    res.add("P3.S6.successor_coldstart", win6 == force_k,
            f"successor request first verify window {win6}, expect "
            f"{force_k} (a slot successor's first verify keeps its own "
            "first proposal's width -- k_eff semantics, NOT the "
            "predecessor's width and NOT launch_k; the pre-1b "
            "launch-width carrier would show launch_k here)")
    res.data["finish_check"] = {"second_request_ok": bool(r6)}
    return res


# ---------------------------------------------------------------------------
# P4 - token-ID comparison vs MTP=3 baseline + stochastic gate
# ---------------------------------------------------------------------------
def _agreement(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    n = max(len(a), len(b))
    same = sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i])
    return same / n


def run_p4(ctx: GateContext, res: PointResult, baseline: dict | None) -> PointResult:
    p4 = ctx.cfg["p4"]
    _assert_boot_phase(ctx, "P4")
    if not baseline or not baseline.get("greedy"):
        res.add("P4.D0.baseline", False,
                "no baseline token-ID capture available; run capture-baseline "
                "on an MTP=3/mode=off boot first (see README)")
        res.add("P4.D1.greedy_token_identity", True, "skipped: no baseline", SKIP)
        res.add("P4.D2.seeded_reproducible", True, "skipped with D0", SKIP)
        res.add("P4.D3.cross_config_agreement", True, "skipped: no baseline", SKIP)
        res.add("P4.D4.distribution_gate", True, "skipped: no baseline", SKIP)
        return res
    res.add("P4.D0.baseline", True,
            f"baseline captured_at={baseline.get('captured_at')} "
            f"greedy={len(baseline['greedy'])} sampled={len(baseline.get('sampled', {}))}")

    # D1 greedy token-ID identity (the primary exactness assert; token IDs, not
    # text bytes)
    mismatches = []
    for key, prompt in P4_PROMPTS:
        ids = ctx.server.chat_token_ids(prompt, max_tokens=64, temperature=0.0)
        base_ids = baseline["greedy"].get(key)
        if base_ids is None:
            mismatches.append((key, "missing-in-baseline"))
            continue
        if ids != base_ids:
            i = next((j for j in range(min(len(ids), len(base_ids)))
                      if ids[j] != base_ids[j]), min(len(ids), len(base_ids)))
            mismatches.append((key, f"first diff at {i}: "
                                     f"got {ids[i:i + 3]}, base {base_ids[i:i + 3]}"))
    res.add("P4.D1.greedy_token_identity", not mismatches,
            "greedy token IDs identical to the MTP=3 baseline on all prompts"
            if not mismatches else f"mismatches: {mismatches}")

    # D2 seeded reproducibility within this boot
    dup_bad = []
    for seed in p4["seeds"][:3]:
        a = ctx.server.chat_token_ids(P4_PROMPTS[0][1],
                                      max_tokens=p4["sample_max_tokens"],
                                      temperature=p4["sample_temperature"], seed=seed)
        b = ctx.server.chat_token_ids(P4_PROMPTS[0][1],
                                      max_tokens=p4["sample_max_tokens"],
                                      temperature=p4["sample_temperature"], seed=seed)
        if a != b:
            dup_bad.append(seed)
    res.add("P4.D2.seeded_reproducible", not dup_bad,
            "same seed -> same token IDs within the forced-K boot"
            if not dup_bad else f"seeds not reproducible: {dup_bad}")

    # D3 cross-config seeded agreement (expect exact; tolerate float drift)
    agreements, fails = {}, []
    for seed in p4["seeds"]:
        ids = ctx.server.chat_token_ids(P4_PROMPTS[0][1],
                                        max_tokens=p4["sample_max_tokens"],
                                        temperature=p4["sample_temperature"], seed=seed)
        base_ids = baseline.get("sampled", {}).get(str(seed))
        if base_ids is None:
            fails.append((seed, "missing-in-baseline"))
            continue
        agreements[str(seed)] = _agreement(ids, base_ids)
        if agreements[str(seed)] < p4["agreement_min"]:
            fails.append((seed, round(agreements[str(seed)], 4)))
    res.data["agreements"] = agreements
    res.add("P4.D3.cross_config_agreement", not fails,
            f"per-seed agreement vs baseline (min {p4['agreement_min']}): "
            + json.dumps({k: round(v, 4) for k, v in agreements.items()})
            + (f"; failures: {fails}" if fails else ""))

    # D4 distribution gate: per-position acceptance-rate delta vs the baseline
    # profile recorded at capture time (catches the stale-proposal-distribution
    # defect class that greedy identity can miss: q sampled vs stale q').
    # Keys normalized to str on BOTH sides: an in-process capture carries int
    # position keys, a baseline loaded through the recorded
    # cfg p4.baseline_file path carries the
    # JSON round-trip's string keys -- position "3" of a K=4 capture must be
    # found either way, never silently defaulted to 0.0 (which would turn the
    # matched-control comparison into a spurious FAIL).
    base_prof = {str(k): float(v)
                 for k, v in (baseline.get("accept_profile") or {}).items()}
    if not base_prof:
        res.add("P4.D4.distribution_gate", True,
                "baseline accept_profile missing (older capture); rerun "
                "capture-baseline to enable D4", WARN)
        return res
    ctx.pair_mark()
    ctx.server.chat([{"role": "user", "content": P4_PROMPTS[0][1]}],
                    max_tokens=p4["sample_max_tokens"],
                    temperature=p4["sample_temperature"], seed=p4["seeds"][0])
    snap1 = ctx.metrics.fetch()
    prev = ctx.pair_prev()
    drafts = delta(prev, snap1, C_DRAFTS)
    if drafts <= 0:
        res.add("P4.D4.distribution_gate", True, "no drafts in the D4 window", WARN)
        return res
    prof = {pos: v / drafts
            for pos, v in pos_delta(prev, snap1, C_ACCEPTED_POS,
                                    range(ctx.launch_k)).items()}
    worst = max(abs(prof.get(p, 0.0) - base_prof.get(str(p), 0.0))
                for p in range(ctx.launch_k))
    res.data["accept_profile_this"] = prof
    res.data["accept_profile_baseline"] = base_prof
    res.add("P4.D4.distribution_gate", worst <= p4["rate_delta_max"],
            f"max per-position acceptance-rate delta vs baseline {worst:.4f} "
            f"(tolerance {p4['rate_delta_max']}); this={prof} base={base_prof}")

    # ---- round 8 (D-014 conditions 1+2): the fingerprinted boot + the
    # C2-level committed-prefix identity ----
    _awfp_boot_check(ctx, res, "P4")

    # FP cross-boot policy: the cache/kernel-selection policy hash of THIS
    # boot vs the baseline boot's recorded policy_sha. A mismatch is the
    # autotune boot-lottery (R-029) -- reported loudly. A baseline captured
    # before the fingerprint layer existed (no awfp_policy_sha) is a WARN,
    # never a silent pass.
    # Re-collect at CHECK time: the boot-time cache holds the EARLY record
    # (collected right after the 4-token probe, before scheduled-step 32
    # settles the policy). By the time this check runs, the workloads have
    # produced the settled record and the comparison must see IT (the
    # cache-first consumer was the reviewer's settled-fingerprint defect).
    # An early!=settled divergence is itself within-boot policy drift and
    # is reported in the detail either way.
    # read the boot-time cache BEFORE re-collecting: the collector
    # overwrites the registry entry, so reading it afterwards would compare
    # the settled value against itself and never detect drift (the
    # reviewer's defect on the first fix)
    early = AWFP_POLICY_BY_BOOT.get(ctx.boot_name)
    settled = _awfp_collect_policy(ctx.logs, ctx.boot_ts, ctx.boot_name)
    this_sha = settled or early
    step_note = ""
    st = AWFP_POLICY_STEP_BY_BOOT.get(ctx.boot_name)
    if settled is None and early is not None:
        # a failed settled collection must not silently degrade into an
        # early-only comparison: the failure note outranks the step stamp
        # (the reviewer caught the if/elif order hiding it whenever a
        # cached step existed)
        step_note = " (settled collection FAILED; compared against the " \
                    "cached early record)"
    elif st is not None:
        step_note = f" (step {st})"
    elif settled is not None:
        # a fresh record with no parseable step must not inherit stale
        # provenance from an earlier collection
        step_note = " (settled; step not recorded)"
    drift_note = ""
    if settled and early and settled != early:
        drift_note = (f" [within-boot policy drift: early {early[:12]}... "
                      f"!= settled {settled[:12]}...]")
    base_sha = baseline.get("awfp_policy_sha")
    base_k = baseline.get("launch_k")
    if base_sha and base_k is not None and int(base_k) != int(ctx.launch_k):
        res.add("P4.FP.cross_boot_policy", True,
                f"shape-mismatched pair (baseline launch_k={base_k} vs this "
                f"{ctx.launch_k}): the policy comparison belongs to the "
                "same-shape matched controls (the revised contract's "
                "off@K4 vs force4); skipped with the reason stated", SKIP)
    elif not this_sha:
        res.add("P4.FP.cross_boot_policy", False,
                "this boot has no AWFP-POLICY record (the fingerprint layer "
                "is absent or never settled) - the cross-boot policy "
                "comparison cannot run", MISSING_INSTRUMENT)
    elif not base_sha:
        res.add("P4.FP.cross_boot_policy", True,
                f"this boot policy_sha={this_sha[:16]}...{step_note}"
                f"{drift_note}; the "
                "baseline carries no policy record (captured before the "
                "fingerprint layer) - rerun capture-baseline on a "
                "fingerprinted floor for the full comparison", WARN)
    else:
        res.add("P4.FP.cross_boot_policy", this_sha == base_sha,
                f"cache/kernel-selection policy (settled): "
                f"this {this_sha[:16]}...{step_note} vs "
                f"baseline {base_sha[:16]}...{drift_note} "
                + ("(identical: no boot-lottery)" if this_sha == base_sha
                   else "MISMATCH - the autotune boot-lottery fired; the "
                        "cross-boot comparison ran on different kernel "
                        "selections (pin: VLLM_TRITON_FORCE_FIRST_CONFIG)"),
                FAIL if this_sha != base_sha else PASS)

    # D5 committed-prefix identity: the C2 record's per-request prefix-hash
    # chain is the ENGINE-SIDE byte-identity signal (prompt + committed
    # output, int64-le sha256). Same-shape matched controls must produce
    # IDENTICAL chains; a shape-mismatched baseline (the legacy MTP=3 floor
    # vs a K=4 gate boot) is skipped with the reason stated -- the revised
    # contract's quantity A belongs to the same-shape off@K4 vs force4 pair.
    base_chains = baseline.get("c2_chains") or {}
    base_k = baseline.get("launch_k")
    if not base_chains:
        res.add("P4.D5.prefix_hash_identity", True,
                "baseline c2_chains missing (older capture); rerun "
                "capture-baseline on a fingerprinted floor to enable D5",
                WARN)
    elif base_k is not None and int(base_k) != int(ctx.launch_k):
        res.add("P4.D5.prefix_hash_identity", True,
                f"shape-mismatched pair (baseline launch_k={base_k} vs this "
                f"{ctx.launch_k}): the committed-prefix identity compares "
                "only same-shape matched controls (the revised contract's "
                "off@K4 vs force4); skipped with the reason stated", SKIP)
    else:
        d5_bad, d5_common = [], 0
        for key, prompt in P4_PROMPTS:
            base_chain = base_chains.get(key) or []
            if not base_chain:
                continue
            t0 = ctx.now()
            _, resp_id = ctx.server.chat_token_ids_tracked(prompt, max_tokens=64,
                                                           temperature=0.0)
            t1 = ctx.now()
            try:
                recs = split_src(ctx.evidence(since=t0, until=t1), "sched")
                chain = _chain_for_response(_chains_in_window(recs), resp_id)
                if chain is None:
                    fresh = _first_appearance_chains(recs)
                    chain = (max(fresh.items(),
                                 key=lambda kv: (len(kv[1]), kv[0]))[1]
                             if fresh else None)
            except Exception as exc:  # noqa: BLE001
                d5_bad.append((key, f"chain collection failed: {exc!r}"))
                continue
            if not chain:
                d5_bad.append((key, "no committed-prefix rows in the D5 "
                                    "window (the C2 commit fields are "
                                    "absent or the request never committed)"))
                continue
            cmp = compare_prefix_chains(base_chain, chain)
            d5_common += cmp["common_ol"]
            if not cmp["match"]:
                d5_bad.append((key, f"common ol {cmp['common_ol']}, first "
                                    f"mismatch {cmp['mismatches'][:1]}"))
        res.add("P4.D5.prefix_hash_identity", not d5_bad and d5_common > 0,
                f"committed-prefix chains: {d5_common} common output lengths "
                "byte-identical to the baseline (prompt + committed output)"
                + (f"; failures: {d5_bad[:2]}" if d5_bad else ""))
    return res


# ---------------------------------------------------------------------------
# P5 - armed-drive verification
# ---------------------------------------------------------------------------
def run_p5(ctx: GateContext, res: PointResult) -> PointResult:
    _assert_boot_phase(ctx, "P5")
    if not _require_instrument(ctx, res, "P5"):
        return res

    # E1 armed path live (install + driving log lines)
    lines = [l for _, l in ctx.logs.since(ctx.boot_ts) if INSTALL_SNIPPET in l]
    armed_line = any(ARMED_SNIPPET in l for _, l in ctx.logs.since(ctx.boot_ts))
    res.add("P5.E1.armed_live", bool(lines) and armed_line,
            f"install lines: {len(lines)}; armed-driving line present: {armed_line}")

    outs = run_waves(ctx.server, ctx.cfg["workload"], sleep=ctx.sleep)
    res.data["waves"] = outs

    series = _window_series(ctx, ctx.boot_ts)
    sched, worker = series["sched"], series["worker"]
    ks = [s.get("requested_k") for s in sched if s.get("requested_k") is not None]
    flips = sum(1 for a, b in zip(ks, ks[1:]) if a != b)

    # E2 the armed decision reaches the DRAFTER (joined by seq). Scheduler-side
    # oscillation with a pinned drafter width is exactly the pre-stage-1b
    # defect state and must FAIL with that reason.
    k_effs = [w.get("k_eff") for w in worker if w.get("k_eff") is not None]
    eff_flips = sum(1 for a, b in zip(k_effs, k_effs[1:]) if a != b)
    pairs, _, _ = join_by_seq(
        [s for s in sched if s.get("requested_k") is not None],
        [w for w in worker if w.get("k_eff") is not None])
    follow = sum(1 for s, w in pairs if s.get("requested_k") == w.get("k_eff"))
    res.data["requested_k_head"] = ks[:16]
    res.data["k_eff_head"] = k_effs[:16]
    if flips == 0:
        res.add("P5.E2.drive_reaches_drafter", False,
                "no requested_k oscillation observed at all - armed path inert")
    elif eff_flips == 0 and pairs and follow < len(pairs):
        res.add("P5.E2.drive_reaches_drafter", False,
                f"scheduler oscillates (flips={flips}) but the drafter width is "
                f"unchanged (k_eff head={k_effs[:10]}) - stage-1b consumption MISSING")
    else:
        res.add("P5.E2.drive_reaches_drafter", bool(pairs) and follow == len(pairs),
                f"requested_k flips={flips}, k_eff flips={eff_flips}, "
                f"follow {follow}/{len(pairs)}")

    # E3 lag consistency during oscillation
    _lag_check(sched, res, "P5.E3.lag_consistency")

    # E4 EMA-pure probe (optional separate boot; soft)
    if ctx.cfg.get("p5", {}).get("run_ema_probe"):
        res.add("P5.E4.ema_pure", True,
                "EMA-pure probe is a separate boot; see README 'what remains manual'",
                WARN)
    else:
        res.add("P5.E4.ema_pure", True,
                "not configured (run_ema_probe=false); armed oscillation is the "
                "primary drive evidence per D-009", SKIP)
    return res


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
# Round 7: the EXPLICIT point-to-boot assignment. P4's fixed-K premise
# (byte-identical greedy token IDs vs the off baseline, D1-D3) only holds on
# the FORCED boot: an armed+oscillate boot varies the proposal width per
# step, which is exactly the degree of freedom the exactness checks must not
# have. P1 (direct-work evidence at the forced width) and P3 (boundary
# coverage) share the forced boot; P2 (in-process transitions) and P5
# (armed-drive verification) need the armed-oscillation boot. The mapping is
# data: run_gate walks gate_boot_plan(), every point runner ASSERTS the boot
# it was handed (_assert_boot_phase), and the selftest asserts the plan
# itself - a wrong assignment can no longer be silent.
POINT_PHASES: dict[str, str] = {
    "P1": "force3",
    "P3": "force3",
    "P4": "force3",
    "P2": "osc",
    "P5": "osc",
}


def _assert_boot_phase(ctx: GateContext, point: str) -> None:
    """Assert the point ran inside its assigned boot phase (round 7).

    A no-op when the context carries no boot name (older callers); the live
    orchestrator and the selftest always set one, so every selftest case
    exercises the assignment too."""
    want = POINT_PHASES[point]
    got = getattr(ctx, "boot_name", None)
    if got is None:
        return
    assert got == want, (
        f"{point} is assigned to the {want!r} boot phase (POINT_PHASES) but "
        f"ran on {got!r} - the point-to-boot assignment is asserted, never "
        "silent (round 7)")


def gate_boot_plan(cfg: dict) -> list[tuple[str, BootSpec, tuple[str, ...]]]:
    """The assert-guarded point-to-boot plan (round 7).

    Returns [(phase_name, spec, points)] in execution order and refuses (by
    assertion) any plan that double-books a point, misses one, or pairs a
    point with a boot whose mode contradicts the point's premise: P4's phase
    must be the FORCED boot at the configured force_k (the fixed-K premise
    D1-D4 compare against), and P2/P5's phase must be the armed-oscillation
    boot (their subject IS the oscillation)."""
    _spec_baseline, spec_force, spec_osc = make_boot_specs(cfg)
    force_k = str(int(cfg["boot"]["force_k"]))
    assert spec_force.aw_mode == "force" and spec_force.aw_force == force_k, (
        f"the force3 spec must be mode=force FORCE={force_k} for P1/P3/P4 "
        f"(got mode={spec_force.aw_mode} force={spec_force.aw_force}) - "
        "P4's fixed-K exactness premise requires the forced boot")
    assert spec_osc.aw_mode == "armed" and spec_osc.aw_force == "oscillate", (
        f"the osc spec must be mode=armed FORCE=oscillate for P2/P5 "
        f"(got mode={spec_osc.aw_mode} force={spec_osc.aw_force})")
    plan = [
        ("force3", spec_force, ("P1", "P3", "P4")),
        ("osc", spec_osc, ("P2", "P5")),
    ]
    seen: list[str] = []
    for name, _spec, pts in plan:
        for p in pts:
            assert POINT_PHASES[p] == name, (
                f"{p} maps to phase {POINT_PHASES[p]!r} but the plan places "
                f"it in {name!r}")
            assert p not in seen, f"{p} is scheduled twice in the boot plan"
            seen.append(p)
    assert set(seen) == set(POINT_PHASES), (
        f"the boot plan covers {sorted(seen)} but the point universe is "
        f"{sorted(POINT_PHASES)}")
    return plan


def make_boot_specs(cfg: dict) -> tuple[BootSpec, BootSpec, BootSpec]:
    """The three gate boots (round 5 adds the baseline boot).

    Run sequence (README boot matrix): off/MTP=3 baseline boot ->
    capture-baseline -> force3 boot (P1/P3/P4) -> armed-osc boot (P2/P5) ->
    restore the operator's baseline config. The baseline boot exists so the
    orchestrator can establish the P4 floor itself when the currently running
    container is not a valid off/MTP=3 boot (the 2026-10-14 run found a
    shadow boot in place and P4 then failed D0 with no baseline).
    """
    b = cfg["boot"]
    spec_baseline = BootSpec("baseline", mtp=3, aw_mode=None, aw_force=None,
                             launch_k=3, candidates=[3],
                             image_override=None, sync_scheduling=True,
                             patch_serve=False,
                             base_config_rel=cfg["baseline_config_dir"])
    spec_force = BootSpec("force3", mtp=b["launch_k"], aw_mode="force",
                          aw_force=str(b["force_k"]), launch_k=b["launch_k"],
                          candidates=b["candidates"],
                          image_override=b.get("image_override"),
                          base_config_rel=cfg["base_config_dir"])
    spec_osc = BootSpec("osc", mtp=b["launch_k"], aw_mode="armed",
                        aw_force="oscillate", launch_k=b["launch_k"],
                        candidates=b["candidates"],
                        image_override=b.get("image_override"),
                        base_config_rel=cfg["base_config_dir"])
    return spec_baseline, spec_force, spec_osc


def _verify_install_from_logs(cfg: dict, boot_ts: float) -> bool | None:
    try:
        lines = DockerLogs(cfg["container"]).since(boot_ts)
    except Exception:
        return None
    for _, l in lines:
        if INSTALL_SNIPPET in l:
            return True
    return False


def _baseline_launch_k(cfg: dict, expected_k: int | None = None) -> int:
    """The launch width the P4 baseline floor is validated/profiled at.

    The RECORDED baseline width wins: a capture stamped launch_k=4 (the
    custom off@K4 matched-control floor) validates and profiles a K=4 off
    floor, so the floor and the gate boots can share a shape ("the current
    boot matches it"). Nothing recorded keeps the historical MTP=3 floor;
    an explicit expected_k overrides both."""
    if expected_k is not None:
        return int(expected_k)
    try:
        bf = (cfg.get("p4") or {}).get("baseline_file")
        if bf and Path(bf).exists():
            k = json.loads(Path(bf).read_text()).get("launch_k")
            if k is not None:
                k = int(k)
                if k >= 1:
                    return k
                # a recorded width below 1 is a corrupt record, not a
                # floor: fall through to the default instead of honoring
                # it verbatim (the reviewer's no-lower-bound finding)
    except Exception:  # noqa: BLE001 - an unreadable record keeps the default
        pass
    return 3


def is_valid_baseline_boot(cfg: dict, env: dict | None = None,
                           argv: list[str] | None = None,
                           expected_k: int | None = None) -> tuple[bool, str]:
    """The running container must be the floor boot: adaptive off, launch K
    matching the RECORDED baseline width (MTP=3 when nothing else is
    recorded).

    The adaptive-width mode is read from Config.Env (docker inspect), NOT from
    argv: the wiring's env travels via `docker run -e` and never appears in
    argv (round-1 attack: a shadow boot validated as the off floor because
    only argv was grepped). MODE=off or unset passes; shadow/armed/force (or
    any other value) is rejected with the observed value named. MTP is read
    from env (MTP=) and argv (num_speculative_tokens) and both must agree at
    the baseline's width when present: the expected width is the recorded
    baseline's stamped launch_k (cfg p4.baseline_file -- a baseline captured
    at K=4 validates a K=4 off floor), falling back to the historical MTP=3
    floor when nothing is recorded; expected_k overrides both."""
    try:
        if env is None:
            env = container_env(cfg["container"])
        if argv is None:
            argv = container_argv(cfg["container"])
    except Exception as e:  # noqa: BLE001
        return False, f"cannot inspect container: {e}"
    mode = env.get("VLLM_ADAPTIVE_WIDTH_MODE")
    if mode is not None and mode.strip().lower() != "off":
        return False, (f"adaptive width mode is {mode!r} in Config.Env "
                       "(not the off floor; the round-1 review landed this "
                       "hole via an argv-only grep)")
    argv_text = " ".join(str(a) for a in argv)
    if mode is None:
        m_mode = re.search(r"VLLM_ADAPTIVE_WIDTH_MODE=([A-Za-z]*)", argv_text)
        if m_mode:
            # a launcher that inlines the env into argv: an explicit off value
            # passes, anything else (or a bare mention) does not
            if m_mode.group(1).strip().lower() != "off":
                return False, (f"adaptive width mode {m_mode.group(1)!r} in "
                               "argv (not the off floor)")
        elif "VLLM_ADAPTIVE_WIDTH_MODE" in argv_text:
            return False, ("adaptive width env present in argv without an "
                           "off value")
    ks = []
    m = re.search(r'"num_speculative_tokens":\s*(\d+)', argv_text) \
        or re.search(r"num_speculative_tokens[\"':= ]+(\d+)", argv_text)
    if m:
        ks.append(("argv", int(m.group(1))))
    if str(env.get("MTP", "")).strip().isdigit():
        ks.append(("env", int(str(env["MTP"]).strip())))
    if not ks:
        return False, "no speculative config found in env or argv"
    expect_k = _baseline_launch_k(cfg, expected_k)
    for src, k in ks:
        if k != expect_k:
            return False, (f"launch K is {k} ({src}), expected {expect_k} "
                           "for the baseline floor (the recorded baseline "
                           "launch_k)")
    seen = ", ".join(f"{s}={k}" for s, k in ks)
    mode_note = f"mode={mode or 'unset'}" if (mode or "off") == "off" \
        else f"mode={mode}"
    return True, f"launch-K={expect_k} baseline confirmed ({seen}, {mode_note})"


def validate_booted_baseline(cfg: dict, env: dict | None = None,
                             argv: list[str] | None = None) -> tuple[bool, str]:
    """Post-boot validation of a HARNESS-booted baseline floor (round 6).

    The round-5 auto-baseline captured the P4 floor without re-validating the
    boot it had just brought up: had the boot come up wrong (armed mode
    inherited from the copied env, the sync EXTRA append lost, a wrong config
    dir), the floor would have been silently poisoned and every D1-D4
    comparison with it. A poisoned floor must FAIL loudly, never silently
    compare. Three conditions: the off/MTP=3 floor (is_valid_baseline_boot:
    mode read from Config.Env, MTP from env AND argv), AND the
    sync-scheduling exclusion the baseline BootSpec demanded actually reached
    the container argv (the env EXTRA append is the only thing that makes the
    floor boot sync; without it the P4 comparison is cross-mode)."""
    ok, why = is_valid_baseline_boot(cfg, env=env, argv=argv)
    if not ok:
        return False, why
    if argv is None:
        try:
            argv = container_argv(cfg["container"])
        except Exception as e:  # noqa: BLE001
            return False, f"cannot inspect container argv: {e}"
    if NO_ASYNC_FLAG not in [str(a).strip() for a in argv]:
        return False, (f"the boot carries no {NO_ASYNC_FLAG} in its argv "
                       "(the env EXTRA append that makes the floor boot SYNC "
                       "did not reach the container; the P4 comparison would "
                       "be cross-mode)")
    return True, (f"off/MTP={_baseline_launch_k(cfg)}/sync floor "
                  f"confirmed ({why})")


def capture_baseline_outputs(server: ServerClient, cfg_p4: dict,
                             logs: LogSource | None = None,
                             clock=None) -> dict:
    """Capture the P4 baseline: greedy token IDs, seeded samples, the
    per-position acceptance profile and -- round 8 -- the C2 committed-
    prefix chains per greedy prompt (the D5 identity comparison's floor).

    The chain capture needs a LOG SOURCE plus a CLOCK matching the server's
    request window (the live Docker logs + wall clock; the selftest's mock
    world + world clock). Each greedy prompt runs alone in its own marked
    window; the request that first commits inside the window is the
    capture's request (the sequential protocol guarantees exactly one).
    """
    out = {"captured_at": time.time(), "greedy": {}, "sampled": {}}
    chains_on = logs is not None and clock is not None
    if chains_on:
        out["c2_chains"] = {}
    for key, prompt in P4_PROMPTS:
        t0 = clock() if chains_on else None
        ids, resp_id = server.chat_token_ids_tracked(prompt, max_tokens=64,
                                                     temperature=0.0)
        out["greedy"][key] = ids
        if chains_on:
            t1 = clock()
            try:
                recs = parse_evidence(logs.since(t0))
                win_chains = _chains_in_window(split_src(recs, "sched"))
                chain = _chain_for_response(win_chains, resp_id)
                if chain is None:
                    # fallback: the round-8 first-appearance protocol (its
                    # ol=0 requirement holds only for chunked prefills)
                    first_app = _first_appearance_chains(split_src(recs, "sched"))
                    chain = (max(first_app.items(),
                                 key=lambda kv: (len(kv[1]), kv[0]))[1]
                             if first_app else [])
                out["c2_chains"][key] = chain
            except Exception:  # noqa: BLE001 - chains are additive evidence
                out["c2_chains"][key] = []
    for seed in cfg_p4["seeds"]:
        out["sampled"][str(seed)] = server.chat_token_ids(
            P4_PROMPTS[0][1], max_tokens=cfg_p4["sample_max_tokens"],
            temperature=cfg_p4["sample_temperature"], seed=seed)
    return out


def _baseline_accept_profile(cfg: dict, expected_k: int | None = None) -> dict | None:
    """Small probe on the running baseline boot: per-position acceptance rates.

    The profile covers the boot's own verified-window positions 0..K-1 --
    the same launch-K range the D4 gate enumerates on the comparison side
    (range(ctx.launch_k)) and the engine's per-pos counters carry
    (range(num_spec_tokens)), parameterized by the recorded baseline width
    via _baseline_launch_k -- never a hard-coded K=3 label set, which
    silently dropped position 3 of a K=4 capture (D4 then compared it
    against the 0.0 default)."""
    try:
        m = LiveMetrics(cfg["base"])
        prev = m.fetch()
        ServerClient(cfg["base"], cfg["model"]).chat(
            [{"role": "user", "content": P4_PROMPTS[0][1]}],
            max_tokens=int(cfg["p4"]["sample_max_tokens"]),
            temperature=float(cfg["p4"]["sample_temperature"]),
            seed=int(cfg["p4"]["seeds"][0]))
        snap = m.fetch()
        drafts = delta(prev, snap, C_DRAFTS)
        if drafts <= 0:
            return None
        width = _baseline_launch_k(cfg, expected_k)
        return {p: v / drafts
                for p, v in pos_delta(prev, snap, C_ACCEPTED_POS,
                                      range(width)).items()}
    except Exception:
        return None


def run_gate(cfg: dict, points: set[str], out_dir: Path) -> int:
    if not cfg.get("allow_boot"):
        print("REFUSING: gate execution boots servers (replacing the running "
              "container). Pass --allow-boot after coordinating with the "
              "operator. The current session must not restart the server.")
        return 2
    scratch = Path(cfg["scratch_parent"]) / "gate-stage1b-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    results: list[PointResult] = []
    baseline = None
    spec_baseline = make_boot_specs(cfg)[0]
    plan = gate_boot_plan(cfg)  # round 7: explicit, assert-guarded assignment
    booted = False

    def _boot(spec: BootSpec) -> GateContext:
        nonlocal booted
        booted = True
        cfg_dir = prepare_config_dir(spec, scratch, Path(cfg["repo_root"]))
        log_path = out_dir / f"boot-{spec.name}.log"
        print(f"[boot] {spec.name}: config dir {cfg_dir}; launching "
              "(replaces the running container)")
        t_launch = time.time() - 5.0  # log horizon: install lines fire during
        # engine init, minutes BEFORE readiness; the log window must start at
        # launch, not at ready
        launch_config_dir(cfg_dir, log_path)
        ready = wait_ready(cfg["base"], float(cfg["boot"]["ready_timeout_s"]))
        print(f"[boot] {spec.name}: ready in {ready:.0f}s")
        ctx = GateContext(
            cfg=cfg, metrics=LiveMetrics(cfg["base"]),
            logs=DockerLogs(cfg["container"]),
            server=ServerClient(cfg["base"], cfg["model"]),
            out_dir=out_dir, boot_ts=t_launch, launch_k=spec.launch_k,
            argv=container_argv(cfg["container"]),
            install_log_ok=_verify_install_from_logs(cfg, t_launch),
            boot_name=spec.name)
        if spec.patch_serve:
            # Round 7: the instrumentation probe. The C2 emitters are
            # REQUEST-DRIVEN (sched: per scheduled step; worker: per propose
            # with requested_k > 0 - the round-5 gate that removed the
            # boot-time JIT-warmup lines), so a FRESH boot has zero evidence
            # lines until the first request runs. The round-7 gate run's
            # P1/P2/P5 A0 checked BEFORE any workload and false-FAILED all
            # three points MISSING_INSTRUMENT on a healthy boot. One tiny
            # request per gate boot guarantees >= 1 scheduled spec step
            # before any point asserts, which makes "lines since boot" a real
            # instrumentation test (a silent or mis-gated emitter still fails
            # A0 - correctly). The baseline boot (patch_serve=False; the v16b
            # image carries neither emitters nor plumbing) needs no probe.
            try:
                probe = ctx.server.chat(
                    [{"role": "user", "content": "Reply with OK."}],
                    max_tokens=4, temperature=0.0)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"[boot] {spec.name}: the instrument probe request failed "
                    f"({e}) - the boot passed readiness but cannot serve; "
                    "refusing to run points against it") from e
            ptoks = ((probe or {}).get("usage") or {}).get("prompt_tokens")
            print(f"[boot] {spec.name}: instrument probe ok "
                  f"(prompt_tokens={ptoks}) - the C2 emitters are "
                  "request-driven; one scheduled step must exist before A0")
        # Round 8: register this boot's settled policy fingerprint so the
        # P4.FP cross-boot comparison can match it against the baseline's.
        _awfp_collect_policy(ctx.logs, ctx.boot_ts, spec.name)
        return ctx

    try:
        if "P4" in points:
            bf = cfg["p4"].get("baseline_file")
            if bf and Path(bf).exists():
                baseline = json.loads(Path(bf).read_text())
            else:
                ok, why = is_valid_baseline_boot(cfg)
                if ok:
                    print("[baseline] capturing the P4 baseline from the "
                          f"running MTP={_baseline_launch_k(cfg)} boot")
                    base_logs = DockerLogs(cfg["container"])
                    baseline = capture_baseline_outputs(
                        ServerClient(cfg["base"], cfg["model"]), cfg["p4"],
                        logs=base_logs, clock=time.time)
                    prof = _baseline_accept_profile(cfg)
                    if prof:
                        baseline["accept_profile"] = prof
                    # Round 8: the floor's shape + fingerprints ride the
                    # capture file (the D5 shape guard and the P4.FP
                    # cross-boot comparison consume them). Stamp the
                    # VALIDATED width, not a hard-coded 3: is_valid_
                    # baseline_boot just accepted the boot AT this width
                    # (a recorded-K4 floor captured here must carry 4, or
                    # the next gate run would reject its own baseline).
                    baseline["launch_k"] = _baseline_launch_k(cfg)
                    base_policy = _awfp_collect_policy(
                        base_logs, time.time() - 3600.0, "baseline")
                    if base_policy:
                        baseline["awfp_policy_sha"] = base_policy
                    if cfg["p4"].get("baseline_file"):
                        Path(cfg["p4"]["baseline_file"]).write_text(
                            json.dumps(baseline, indent=1))
                else:
                    # Round 5 (documented run sequence): establish the floor
                    # OURSELVES instead of failing P4.D0 -- boot the
                    # off/MTP=3 baseline config (a guarded scratch copy of
                    # baseline_config_dir via the spec's base_config_rel,
                    # sync scheduling like every gate boot so the P4
                    # comparison stays same-mode), capture, then continue
                    # into the gate boots. The final restore still reboots
                    # the operator's UNMODIFIED baseline copy.
                    print(f"[baseline] the running boot is not the MTP=3/off "
                          f"floor ({why}); booting the baseline config first "
                          f"(copy of {cfg['baseline_config_dir']}; documented "
                          "sequence: off/MTP=3 boot -> "
                          "capture-baseline -> gate boots -> restore)")
                    base_ctx = _boot(spec_baseline)
                    # Round 6: re-validate the boot BEFORE capturing. The
                    # round-5 path captured whatever came up with no checks
                    # -- a poisoned floor (armed mode, lost sync flag) would
                    # have silently poisoned every D1-D4 comparison.
                    ok2, why2 = validate_booted_baseline(cfg)
                    if not ok2:
                        print(f"[baseline] REFUSING to capture: {why2}")
                        raise RuntimeError(
                            "the freshly booted baseline is NOT a valid "
                            f"off/MTP={_baseline_launch_k(cfg)}/sync floor "
                            f"({why2}) - a poisoned "
                            "floor must fail loudly, never silently poison "
                            "the P4 comparison (round 6)")
                    baseline = capture_baseline_outputs(
                        ServerClient(cfg["base"], cfg["model"]), cfg["p4"],
                        logs=base_ctx.logs, clock=time.time)
                    prof = _baseline_accept_profile(cfg)
                    if prof:
                        baseline["accept_profile"] = prof
                    # Round 8: the floor's shape + fingerprints ride the
                    # capture file (the D5 shape guard and the P4.FP
                    # cross-boot comparison consume them).
                    baseline["launch_k"] = int(spec_baseline.launch_k)
                    base_policy = _awfp_collect_policy(
                        base_ctx.logs, base_ctx.boot_ts, "baseline")
                    if base_policy:
                        baseline["awfp_policy_sha"] = base_policy
                    if cfg["p4"].get("baseline_file"):
                        Path(cfg["p4"]["baseline_file"]).write_text(
                            json.dumps(baseline, indent=1))
                    print("[baseline] captured from the freshly booted "
                          "off/MTP=3 floor")
        # Round 7: the phases are walked from the assert-guarded plan (see
        # gate_boot_plan); each runner re-asserts the boot it was handed via
        # _assert_boot_phase. Point order within a phase is unchanged
        # (force3: P1, P3, P4 - osc: P2, P5).
        for _phase_name, spec, phase_points in plan:
            if not points & set(phase_points):
                continue
            ctx = _boot(spec)
            for p in ("P1", "P3", "P4", "P2", "P5"):
                if p not in points or p not in phase_points:
                    continue
                if p == "P1":
                    results.append(run_p1(ctx, PointResult("P1")))
                elif p == "P3":
                    results.append(run_p3(ctx, PointResult("P3")))
                elif p == "P4":
                    results.append(run_p4(ctx, PointResult("P4"), baseline))
                elif p == "P2":
                    results.append(run_p2(ctx, PointResult("P2")))
                elif p == "P5":
                    results.append(run_p5(ctx, PointResult("P5")))
    finally:
        if booted:
            try:
                dst = scratch / "config-restore-baseline"
                if dst.exists():
                    shutil.rmtree(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(Path(cfg["repo_root"]) / cfg["baseline_config_dir"], dst)
                print("[restore] rebooting the MTP=3 baseline config (v16b copy)")
                launch_config_dir(dst, out_dir / "boot-restore.log")
                wait_ready(cfg["base"], float(cfg["boot"]["ready_timeout_s"]))
                print("[restore] baseline restored")
            except Exception as e:  # noqa: BLE001
                print(f"[restore] FAILED: {e} - restore the baseline manually "
                      f"(copy of {cfg['baseline_config_dir']}, launch.sh --run)")

    report = {
        "harness": "gate-stage1b",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "points": [r.to_json() for r in results],
        "overall": _overall(results),
    }
    out_path = out_dir / f"gate-stage1b-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=1))
    _print_table(results)
    print(f"result file: {out_path}")
    return 0 if report["overall"] == PASS else 1


def _overall(results: list[PointResult]) -> str:
    sts = [r.status for r in results]
    if any(s in (FAIL, MISSING_INSTRUMENT) for s in sts):
        return FAIL
    if any(s == WARN for s in sts):
        return WARN
    if not sts:
        return SKIP
    return PASS


def _print_table(results: list[PointResult]) -> None:
    print("\n=== gate-stage1b results ===")
    for r in results:
        print(f"{r.point}: {r.status}")
        for c in r.checks:
            print(f"  [{c.status:>19}] {c.id}: {c.detail}")
    print(f"OVERALL: {_overall(results)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_config(path: Path | None) -> dict:
    return json.loads((path or DEFAULT_CONFIG).read_text())


def _default_out_dir(cfg: dict) -> Path:
    out = cfg.get("out_dir")
    if out:
        return Path(out)
    return Path(cfg["scratch_parent"]) / "2026-10-13-p2-stage1b-gate"


def check_live(cfg: dict) -> int:
    """READ-ONLY: scrape /metrics once, inspect argv, report the inventory."""
    base = cfg["base"]
    print(f"scraping {base}/metrics (read-only; no workload, no boot)")
    try:
        snap = LiveMetrics(base).fetch()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot scrape: {e}")
        return 1
    for n in (C_DRAFTS, C_DRAFT_TOKENS, C_ACCEPTED, C_ACCEPTED_POS, C_SAMPLES_POS):
        if n in snap:
            labels = sorted({json.dumps(l, sort_keys=True) for l, _ in snap[n]})
            print(f"  present: {n} ({len(snap[n])} samples)")
            for l in labels[:3]:
                print(f"    labels: {l}")
        else:
            print(f"  ABSENT : {n}"
                  + ("  <- stage-1b patch not installed"
                     if n == C_SAMPLES_POS else ""))
    try:
        argv = container_argv(cfg["container"])
        env = container_env(cfg["container"])
        flags = argv_async_flags(argv)
        print(f"  container argv async-scheduling ENABLE flags: {flags or 'none'}"
              " (negations like --no-async-scheduling excluded, round 6;"
              " secondary signal only: SchedulerConfig.async_scheduling=None"
              " AUTO-resolves to True for MTP boots, so a clean argv does NOT"
              " mean async is off -- P3.S7 asserts the sync-channel log line"
              " instead)")
        print(f"  container image: {container_image(cfg['container'])}")
        print(f"  Config.Env adaptive mode: "
              f"{env.get('VLLM_ADAPTIVE_WIDTH_MODE', 'unset')!r} "
              "(read from Config.Env, not argv)")
    except Exception as e:  # noqa: BLE001
        print(f"  docker inspect failed: {e}")
    ok, why = is_valid_baseline_boot(cfg)
    print(f"  baseline-boot check: {'OK' if ok else 'NO'} ({why})")
    return 0


def capture_baseline(cfg: dict) -> int:
    if not cfg.get("live_ok"):
        print("REFUSING: capture-baseline sends requests to the RUNNING server. "
              "Pass --live-ok after coordinating with the operator.")
        return 2
    ok, why = is_valid_baseline_boot(cfg)
    if not ok:
        print(f"REFUSING: the running boot is not the MTP=3 baseline: {why}")
        return 2
    out_dir = _default_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = DockerLogs(cfg["container"])
    server = ServerClient(cfg["base"], cfg["model"])
    baseline = capture_baseline_outputs(server, cfg["p4"], logs=logs,
                                        clock=time.time)
    prof = _baseline_accept_profile(cfg)
    if prof:
        baseline["accept_profile"] = prof
    # Round 8: the baseline boot's fingerprints ride the capture file -- the
    # P4.FP checks compare the arms' policy_sha against it.
    try:
        boot_ts = time.time() - 3600.0  # the running boot predates the capture
        policy = _awfp_collect_policy(logs, boot_ts, "baseline")
        if policy:
            baseline["awfp_policy_sha"] = policy
        recs = parse_awfp(logs.since(boot_ts))
        probes = [r for r in recs if r.get("kind") == "PROBE"]
        if probes:
            baseline["awfp_probe_passed"] = all(
                r.get("passed") is True for r in probes)
    except Exception:  # noqa: BLE001 - fingerprint capture is additive
        pass
    path = Path(cfg["p4"].get("baseline_file") or (out_dir / "baseline-mtp3.json"))
    path.write_text(json.dumps(baseline, indent=1))
    print(f"baseline captured -> {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="stage-1b five-point gate harness")
    ap.add_argument("command", nargs="?", default="selftest",
                    choices=["selftest", "check-live", "capture-baseline", "run"])
    ap.add_argument("--config", default=None, help="gate_config.json path")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--points", default="P1,P2,P3,P4,P5",
                    help="comma list for run: P1,P2,P3,P4,P5")
    ap.add_argument("--allow-boot", action="store_true",
                    help="REQUIRED for run: permits boots that replace the "
                         "running container (restore-baseline runs afterwards)")
    ap.add_argument("--live-ok", action="store_true",
                    help="REQUIRED for capture-baseline: permits sending the "
                         "small P4 prompt set to the RUNNING server")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    if args.allow_boot:
        cfg["allow_boot"] = True
    if args.live_ok:
        cfg["live_ok"] = True

    if args.command == "selftest":
        out_dir = Path(args.out) if args.out else _default_out_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(HERE))
        import gate_selftest
        return gate_selftest.run(out_dir)
    if args.command == "check-live":
        return check_live(cfg)
    if args.command == "capture-baseline":
        return capture_baseline(cfg)
    if args.command == "run":
        points = {p.strip().upper() for p in args.points.split(",") if p.strip()}
        out_dir = Path(args.out) if args.out else _default_out_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        return run_gate(cfg, points, out_dir)
    return 2


if __name__ == "__main__":
    sys.exit(main())
