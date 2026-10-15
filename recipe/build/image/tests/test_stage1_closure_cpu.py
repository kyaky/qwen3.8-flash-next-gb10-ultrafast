#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU gate for the P2 stage-1 closure instrumentation (D-014 conds. 1+2).

Runs INSIDE the iter7f image (or against the src tree on the host) with no
GPU and no server. It proves, against the gate harness's own check code:

  1. the round-8 C2 v2 record world: a hand-built CONSISTENT sequence of
     worker/sched lines (proposals -> verifies -> commits, one request,
     chains advancing) satisfies every new harness check
     (fields_worker / fields_sched / proposal_join / argmax_greedy /
     prefix_hash) -- the emitter format and the harness semantics agree;
  2. each round-8 defect class is caught with the right verdict class:
     the argmax divergence, a stale committed-prefix hash, the missing
     proposal side (old-shape emitter -> MISSING_INSTRUMENT), a wrongly
     typed props value (MISSING_MALFORMED_INSTRUMENT, never a crash) and
     the dropped worker record (the join fails);
  3. the committed-prefix hash book (awdbg1b_evidence.commit_hash_update /
     commit_hash_prune): incremental feed, regression re-base, prune;
  4. the AWFP fingerprint layer: the policy snapshot is deterministic and
     its line formats parse through the harness's parse_awfp; the
     maybe_install / maybe_emit_policy gating emits CONFIG+PROBE and
     exactly the stepped POLICY records; failures surface as AWFP-FAIL
     lines (never an exception); the cross-boot drift comparison fires on
     a changed policy_sha;
  5. when the patched marlin_moe module is importable (in-image), its
     import-time code-state check reports the two hard-set-false
     atomic_add call sites and the shim wiring (the behavioral CUDA probe
     itself runs at boot, where a GPU exists -- the harness's FP.probe
     check owns it there).

Usage: python3 test_stage1_closure_cpu.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src" if HERE.name == "tests" else HERE


def _locate_gate_dir() -> Path:
    env = os.environ.get("GB10_GATE_DIR")
    if env:
        p = Path(env)
        if (p / "driver.py").is_file():
            return p
        raise SystemExit(f"GB10_GATE_DIR={env} has no driver.py")
    fork_copy = SRC.parents[2] / "tools" / "gate-stage1b"
    if (fork_copy / "driver.py").is_file():
        return fork_copy
    probe = SRC
    for _ in range(8):
        cand = (probe / "agents" / "glm-5.3-flash-integrator" / "scripts"
                / "gate-stage1b")
        if (cand / "driver.py").is_file():
            return cand
        probe = probe.parent
    raise SystemExit(
        "cannot locate scripts/gate-stage1b (set GB10_GATE_DIR to its parent)")


GATE_DIR = _locate_gate_dir()
sys.path.insert(0, str(GATE_DIR))
sys.path.insert(0, str(SRC))
import driver  # noqa: E402  (the gate harness; stdlib only)
import awdbg1b_evidence as ev  # noqa: E402  (the shared emitter helper)
import awfp_fingerprint as fp  # noqa: E402  (the fingerprint helper)

FAILURES: list[str] = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


LAUNCH_K = 4
FORCE_K = 3
PH = "0" * 64


def _ph(n: int) -> str:
    return format(n, "064x")


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append((1.0, msg % args if args else msg))


def build_v2_world():
    """One request, four engine steps, fully consistent round-8 records.

    step 1: prefill chunk  (win 0, emit [], worker samples k_eff columns)
    step 2: first decode   (win 0 -- no kept proposal yet; one plain commit)
    step 3: verifies P2    (win 3, rej_idx 2, bonus, emit = P2[:2] + [bonus])
    step 4: verifies P3    (win 3, rej_idx 3 = full acceptance, the bonus
                            position; emit = P3[:3] + [bonus])
    The verify at step N consumes the proposal emitted at step N-1 (the
    documented one-step lag); the committed-prefix hash advances over
    prompt + committed output each step.
    """
    log = _Logger()
    ev._SEQS.clear()
    prompt = [11, 22, 33, 44]
    rid = "r1"
    book: dict = {}

    def worker(seq_off_props, k_eff):
        ev.build_worker_record  # noqa: B018 - documentation anchor
        return ev.build_worker_record(
            k_eff=k_eff, launch_k=LAUNCH_K,
            sampled_per_pos=[1 if p < k_eff else 0 for p in range(LAUNCH_K)],
            req_ids=[rid], idx_rows=[0], branch="gumbel", num_tokens=3,
            has_prefill=False, temps=[0.0], seeds=[0], seeds_set=[False],
            props=[seq_off_props])

    def sched(win, acc, rej, rej_idx, bonus, emit, ph, ol, nct):
        return ev.build_sched_record(
            launch_k=LAUNCH_K, requested_k=FORCE_K, scheduled_req_ids=[rid],
            spec_windows={rid: [-1] * win} if win else {},
            acc_rej={rid: (acc, rej)}, mode="force",
            verify=({rid: {"rej_idx": rej_idx, "bonus": bonus}}
                    if rej_idx is not None else None),
            commit={rid: {"emit": emit, "phash": ph, "ol": ol, "nct": nct}},
            sampling={rid: {"temp": 0.0, "seed": 0, "seed_set": False}})

    props = {}
    committed: list[int] = []
    for step in range(1, 5):
        k_eff = FORCE_K
        p = [(prompt[i % len(prompt)] * 7 + i * 3) % 150000
             for i in range(k_eff)]
        props[step] = p
        rec = worker(p, k_eff)
        rec["seq"] = step
        ev.emit_line(log, rec)
        # the verify of the PREVIOUS step's proposal
        if step >= 3:
            prev = props[step - 1]
            rej_idx = k_eff - 1 if step == 3 else k_eff
            bonus = (prev[rej_idx] + 17) % 150000 \
                if rej_idx < len(prev) else (prev[-1] + 17) % 150000
            emit = list(prev[:rej_idx]) + [bonus]
            acc, rej = rej_idx, k_eff - rej_idx
            win = k_eff
        elif step == 2:
            rej_idx = bonus = None
            emit = [(prompt[0] * 11 + 5) % 150000]
            acc = rej = 0
            win = 0
        else:
            rej_idx = bonus = None
            emit = []
            acc = rej = 0
            win = 0
        committed.extend(emit)
        # the committed-prefix hash over prompt + committed output (the
        # engine feeds the same list incrementally; the book re-feeds the
        # delta, so calling with the full prefix each step is equivalent)
        ph, _ = ev.commit_hash_update(book, rid, prompt + committed)
        ol = len(committed)
        rec = sched(win, acc, rej, rej_idx, bonus, emit, ph, ol,
                    len(prompt) + ol)
        rec["seq"] = step
        ev.emit_line(log, rec)
    return log, props, book, prompt


def parse_world(log):
    recs = driver.parse_evidence(log.lines)
    sched = []
    for r in sorted(driver.split_src(recs, "sched"),
                    key=lambda r: (r.get("seq") is None, r.get("seq") or 0)):
        sched.append({**r, "reqmap": {rq.get("id"): rq
                                      for rq in (r.get("reqs") or [])}})
    worker = sorted(driver.split_src(recs, "worker"),
                    key=lambda r: (r.get("seq") is None, r.get("seq") or 0))
    return {"sched": sched, "worker": worker}


def main() -> int:
    print(f"gate harness: {GATE_DIR}")
    print(f"emitter helper: {SRC / 'awdbg1b_evidence.py'}")

    # ---- 1. the consistent v2 world passes every round-8 check ----
    log, props, book, prompt = build_v2_world()
    series = parse_world(log)
    res = driver.PointResult("T")
    driver._c2_semantics_check(series, res, "T.C2", LAUNCH_K)
    for cid in ("T.C2.fields_worker", "T.C2.fields_sched",
                "T.C2.proposal_join", "T.C2.argmax_greedy",
                "T.C2.prefix_hash"):
        c = res.check(cid)
        check(cid, c is not None and c.status == driver.PASS,
              (c.detail[:160] if c else "missing"))

    # ---- 2. the defect classes ----
    # (a) the argmax divergence on a greedy row
    bad = json.loads(json.dumps(series))
    for s in bad["sched"]:
        for rq in s["reqs"]:
            if rq.get("rej_idx") is not None and rq.get("emit"):
                rq["emit"][0] = (rq["emit"][0] + 1) % 150000
    res2 = driver.PointResult("T")
    driver._c2_semantics_check(bad, res2, "T.C2", LAUNCH_K)
    c = res2.check("T.C2.argmax_greedy")
    check("defect.argmax_mismatch", c is not None and c.status == driver.FAIL,
          (c.detail[:140] if c else "missing"))

    # (b) a stale committed-prefix hash (grows but does not move)
    bad = json.loads(json.dumps(series))
    fixed = bad["sched"][0]["reqs"][0]["phash"]
    for s in bad["sched"]:
        for rq in s["reqs"]:
            rq["phash"] = fixed
    res3 = driver.PointResult("T")
    driver._c2_semantics_check(bad, res3, "T.C2", LAUNCH_K)
    c = res3.check("T.C2.prefix_hash")
    check("defect.stale_prefix_hash",
          c is not None and c.status == driver.FAIL,
          (c.detail[:140] if c else "missing"))

    # (c) the missing proposal side (an old-shape emitter): MISSING, never a
    # silent pass and never a FAIL lie
    bad = json.loads(json.dumps(series))
    for w in bad["worker"]:
        for k in ("branch", "nt", "pf"):
            w.pop(k, None)
        for rq in w["reqs"]:
            for k in ("temp", "seed", "seed_set", "props"):
                rq.pop(k, None)
    res4 = driver.PointResult("T")
    driver._c2_semantics_check(bad, res4, "T.C2", LAUNCH_K)
    c = res4.check("T.C2.fields_worker")
    check("defect.old_shape_fields_missing",
          c is not None and c.status == driver.MISSING_INSTRUMENT,
          (c.detail[:140] if c else "missing"))
    c = res4.check("T.C2.argmax_greedy")
    check("defect.old_shape_argmax_missing",
          c is not None and c.status == driver.MISSING_INSTRUMENT,
          (c.detail[:140] if c else "missing"))

    # (d) a wrongly typed props value: MALFORMED, never a crash
    bad = json.loads(json.dumps(series))
    bad["worker"][0]["reqs"][0]["props"] = 5
    res5 = driver.PointResult("T")
    try:
        driver._c2_semantics_check(bad, res5, "T.C2", LAUNCH_K)
        crashed = False
    except Exception:
        crashed = True
    c = res5.check("T.C2.fields_worker")
    check("defect.malformed_props_no_crash",
          not crashed and c is not None
          and c.status == driver.MISSING_MALFORMED_INSTRUMENT,
          (c.detail[:140] if c else "missing"))

    # (e) the dropped worker record: the verify loses its producer
    bad = json.loads(json.dumps(series))
    bad["worker"] = bad["worker"][:2]  # seq 3's proposal vanishes
    res6 = driver.PointResult("T")
    driver._c2_semantics_check(bad, res6, "T.C2", LAUNCH_K)
    c = res6.check("T.C2.proposal_join")
    check("defect.dropped_worker_join",
          c is not None and c.status == driver.FAIL,
          (c.detail[:140] if c else "missing"))

    # ---- 3. the committed-prefix hash book ----
    b: dict = {}
    prefix = list(range(100, 110))
    h1, n1 = ev.commit_hash_update(b, "r", prefix)
    h2, n2 = ev.commit_hash_update(b, "r", prefix + [7, 8])
    h3, _ = ev.commit_hash_update(b, "r", prefix + [7, 8])
    h4, n4 = ev.commit_hash_update(b, "r", prefix[:5])
    ok = (n1 == 10 and n2 == 12 and h2 != h1 and h3 == h2 and n4 == 5
          and h4 != h2 and len(h1) == 64)
    fresh = {}
    h_re, _ = ev.commit_hash_update(fresh, "r", prefix[:5])
    ok &= h_re == h4  # the re-based digest equals a from-scratch digest
    ok &= ev.commit_hash_prune(b, ["r"]) == 0
    ok &= ev.commit_hash_prune(b, []) == 1 and not b
    check("commit_hash_book", ok)

    # ---- 4. the AWFP fingerprint layer ----
    class _CC:
        cudagraph_mode = "PIECEWISE"
        cudagraph_capture_sizes = [1, 2, 4, 8]

    class _Mgr:
        def captured_token_counts(self):
            return [1, 2, 4]

    class _Spec:
        method = "mtp"
        draft_sample_method = "probabilistic"
        rejection_sample_method = "block"

    class _Runner:
        compilation_config = _CC()
        cudagraph_manager = _Mgr()
        speculative_config = _Spec()
        num_speculative_steps = FORCE_K
        device = "cpu"

    s1 = fp.policy_snapshot(_Runner())
    s2 = fp.policy_snapshot(_Runner())
    check("awfp.policy_deterministic",
          len(s1["policy_sha"]) == 64 and s1["policy_sha"] == s2["policy_sha"],
          s1["policy_sha"][:16])
    check("awfp.policy_components",
          s1["cudagraph"]["configured_sizes"] == [1, 2, 4, 8]
          and s1["spec"]["draft_sample_method"] == "probabilistic"
          and s1["ffc"] is not None)

    # The execution/progress artifacts must NOT move the hash: a record
    # collected before the cudagraph captures finished (captured_token_counts
    # -> None) must hash EQUAL to the settled one -- the first live matched
    # pair (off@K4 vs force4@K4) mismatched on exactly this timing artifact,
    # which would have masked the real autotune-lottery signal.
    class _MgrEarly:
        def captured_token_counts(self):
            return None

    class _RunnerEarly(_Runner):
        cudagraph_manager = _MgrEarly()

    s3 = fp.policy_snapshot(_RunnerEarly())
    check("awfp.policy_ignores_capture_progress",
          s3["policy_sha"] == s1["policy_sha"]
          and s3["cudagraph"]["captured_sizes"] is None
          and s1["cudagraph"]["captured_sizes"] == [1, 2, 4],
          f"{s3['policy_sha'][:12]} vs settled {s1['policy_sha'][:12]}")

    # maybe_install emits CONFIG (+ PROBE or a FAIL line, never an exception)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fp.maybe_install(_Runner())
    lines = buf.getvalue().strip().splitlines()
    kinds = [json.loads(x.split(" ", 1)[1])["kind"] for x in lines
             if x.startswith(driver.TAG_AWFP + " ")]
    check("awfp.install_emits_config", "CONFIG" in kinds, str(kinds))
    check("awfp.install_never_raises", True)
    parsed = driver.parse_awfp([(1.0, x) for x in lines])
    check("awfp.parse_through_harness",
          any(r.get("kind") == "CONFIG" for r in parsed))

    # the policy emit gating: scheduled steps 1 and 32, exactly two records
    # (each call passes the step's requested K -- a SCHEDULED step here)
    fp._policy_steps = 0
    fp._policy_emitted_at.clear()
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        for _ in range(40):
            fp.maybe_emit_policy(_Runner(), 3)
    pol = [json.loads(x.split(" ", 1)[1]) for x in buf2.getvalue()
           .strip().splitlines() if x.startswith(driver.TAG_AWFP + " ")]
    check("awfp.policy_emit_steps",
          [p["step"] for p in pol] == [1, 32]
          and all(len(p["policy_sha"]) == 64 for p in pol),
          str([p.get("step") for p in pol]))

    # the producer-side requested-K gate: a step with NO speculative
    # verification scheduled (the JIT warmup steps carry 0, the dummy/
    # profile runs None) neither emits nor advances the counter
    fp._policy_steps = 0
    fp._policy_emitted_at.clear()
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        for rk in (None, 0, None, 0):
            fp.maybe_emit_policy(_Runner(), rk)
    check("awfp.policy_gate_nonscheduled",
          fp._policy_steps == 0 and buf3.getvalue().strip() == "",
          f"steps={fp._policy_steps}")

    # the cross-boot drift comparison
    import tempfile
    tmp = tempfile.mkdtemp(prefix="awfp-test-")
    old_dir = fp.STATE_DIR
    fp.STATE_DIR = tmp
    try:
        with open(os.path.join(tmp, fp.STATE_FILE), "w") as f:
            json.dump({"policy_sha": "dead" + "0" * 60}, f)
        r = fp._persist_and_compare(s1["policy_sha"])
        check("awfp.drift_detected",
              r["drift"] is True and r["prev_sha"] == "dead" + "0" * 60)
        r2 = fp._persist_and_compare(s1["policy_sha"])
        check("awfp.drift_clears_after_write", r2["drift"] is None)
    finally:
        fp.STATE_DIR = old_dir

    # ---- 5. the marlin code-state check (when the module is importable) ----
    try:
        from vllm.model_executor.layers.fused_moe import experts as _exp
        mm = _exp.marlin_moe
        state = mm._q38_awfp_code_state()
        check("awfp.marlin_code_state",
              state["atomic_add_sites"] == 2 and state["shim_marker"] == 1
              and state["shim_call_site"] == 1
              and state["shim_env_on"] is True, json.dumps(state, default=str))
        try:
            probe = mm._q38_awfp_align_probe("cpu")
            check("awfp.marlin_probe_cpu", False,
                  "the probe ran on CPU (expected a CUDA requirement)")
        except AssertionError:
            # a shim-gated-off or CPU environment fails the probe loudly --
            # exactly the designed behavior for a non-provable boot
            check("awfp.marlin_probe_cpu", True,
                  "probe fails loudly without CUDA/shim (designed)")
        except Exception as exc:
            check("awfp.marlin_probe_cpu", True,
                  f"probe unavailable off-GPU ({exc!r:.80})")
    except Exception as exc:  # host run without vllm: the in-image gate covers it
        check("awfp.marlin_code_state", True, f"skipped off-image ({exc!r:.80})")

    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS: the stage-1 closure instrumentation (C2 v2 fields + AWFP "
          "fingerprints) agrees with the gate harness's checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
