#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU self-check for the stage-1b AWDBG1B emitters (P2 stage-1b, round 5).

Proves emitter-format compatibility BEFORE any boot: the synthetic worker and
sched lines are built by the REAL emitter code path (the shared
awdbg1b_evidence helper the image copies into site-packages -- the exact
JSON construction the live scheduler.py / gpu_model_runner.py emitters call),
rendered through the REAL emit_line formatting, then parsed and checked by
the REAL gate harness (scripts/gate-stage1b/driver.py):

  - parse_evidence / split_src / join_by_seq (pairing by seq),
  - _alignment_check  (P1.A8 / P2.B6: row ownership, steps == range(valid_len),
    steps < k_eff, reqs payload required and type-checked, worker row count
    vs sched bs),
  - _lag_check        (P2.B2 / P5.E3: win(t) == requested_k(t-1), trims exempt),
  - _window_bound_violations (P1.A4b, round-4 form: a window can never be
    longer than the previous proposal),
  - the round-4 WORKER-TRUTH contract shape: sampled_per_pos is the
    speculator's executed-sample counter shipped VERBATIM (the C1 list),
    valid_len == k_eff and steps == range(k_eff) on EVERY row (rows without
    a scheduled verify window -- chunked prefill -- consume their samples
    too; their windows stay empty scheduler-side),
  - the P1 contract shape asserts (k_eff, support, no sample at position K,
    uniform executed sampling sum(spp) == bs x k_eff per line),
  - the P2.B4 aggregate identity per paired step,
  - the round-3 malformed-instrumentation contract (reqs omitted -> MISSING_
    INSTRUMENT; a scalar reqs[].steps -> MISSING_MALFORMED_INSTRUMENT, never
    a crash).

Imports NOTHING from vllm (the harness is stdlib-only, and so is the
emitter helper). Exits 0 on PASS, 1 on any failure.

Usage: python3 test_width_runner_cpu.py
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _locate_gate_dir() -> Path:
    """Find the gate harness dir (env override, else: the fork-archived copy
    at recipe/tools/gate-stage1b, else walk up to the workspace root and look
    under agents/glm-5.3-flash-integrator/scripts/gate-stage1b)."""
    env = os.environ.get("GB10_GATE_DIR")
    if env:
        p = Path(env)
        if (p / "driver.py").is_file():
            return p
        raise SystemExit(f"GB10_GATE_DIR={env} has no driver.py")
    fork_copy = HERE.parents[2] / "tools" / "gate-stage1b"
    if (fork_copy / "driver.py").is_file():
        return fork_copy
    probe = HERE
    for _ in range(8):
        cand = (
            probe
            / "agents"
            / "glm-5.3-flash-integrator"
            / "scripts"
            / "gate-stage1b"
        )
        if (cand / "driver.py").is_file():
            return cand
        probe = probe.parent
    raise SystemExit(
        "cannot locate scripts/gate-stage1b (set GB10_GATE_DIR to its parent)"
    )


GATE_DIR = _locate_gate_dir()
sys.path.insert(0, str(GATE_DIR))
import driver  # noqa: E402  (the gate harness; stdlib only)

sys.path.insert(0, str(HERE))
import awdbg1b_evidence as ev  # noqa: E402  (the shared emitter helper)

FAILURES: list[str] = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


class FakeLogger:
    """Captures emit_line output exactly as a vllm logger would render it."""

    def __init__(self):
        self.lines: list[tuple[float, str]] = []

    def info(self, msg, *args):
        rendered = msg % args if args else msg
        self.lines.append((time.time(), rendered))


def emit_worker(fake_log, **kw):
    """Emit one worker record with the round-8 proposal-side fields defaulted
    in (branch/nt/pf + per-row temp/seed/seed_set/props), so every synthetic
    record carries the full v2 C2 contract; explicit kwargs override."""
    req_ids = kw.get("req_ids") or []
    k_eff = int(kw.get("k_eff") or 0)
    kw.setdefault("branch", "gumbel")
    kw.setdefault("num_tokens", max(1, len(req_ids)) * (k_eff or 1))
    kw.setdefault("has_prefill", False)
    if "temps" not in kw:
        kw["temps"] = [0.0] * len(req_ids)
        kw["seeds"] = [0] * len(req_ids)
        kw["seeds_set"] = [False] * len(req_ids)
        kw["props"] = [[(i * 100 + 7 + p) % 150000 for p in range(k_eff)]
                       for i in range(len(req_ids))]
    rec = ev.build_worker_record(**kw)
    ev.emit_line(fake_log, rec)
    return rec


def emit_sched(fake_log, **kw):
    """Emit one sched record with the round-8 verify/commit-side fields
    defaulted in (rej_idx/bonus from the acc counts; emit/phash/ol/nct and
    the sampling mirror per row); explicit kwargs override. The defaults are
    SHAPE-complete, not semantically chained -- the round-8 semantic checks
    have their own consistent world in test_stage1_closure_cpu.py."""
    ids = kw.get("scheduled_req_ids") or []
    windows = kw.get("spec_windows") or {}
    acc_rej = kw.get("acc_rej") or {}
    if "verify" not in kw:
        kw["verify"] = {rid: {"rej_idx": int(acc_rej[rid][0]),
                              "bonus": (acc_rej[rid][0] + 41) % 150000}
                        for rid in ids if rid in acc_rej}
    if "commit" not in kw:
        kw["commit"] = {rid: {"emit": [(i * 13 + 5) % 150000
                                       for _ in range(int(acc_rej[rid][0]) + 1)],
                              "phash": format(i + 1, "064x"),
                              "ol": int(acc_rej[rid][0]) + 1, "nct": 100 + i}
                       for i, rid in enumerate(ids) if rid in acc_rej}
    if "sampling" not in kw:
        kw["sampling"] = {rid: {"temp": 0.0, "seed": 0, "seed_set": False}
                          for rid in ids}
    rec = ev.build_sched_record(**kw)
    ev.emit_line(fake_log, rec)
    return rec


def parse_log(fake_log):
    recs = driver.parse_evidence(fake_log.lines)
    # Wrap the sched records exactly like the harness's _window_series does
    # (the lag/bounds checks consume reqmap-wrapped records).
    sched = []
    for r in sorted(driver.split_src(recs, "sched"),
                    key=lambda r: (r.get("seq") is None, r.get("seq") or 0)):
        sched.append({**r, "reqmap": {rq.get("id"): rq
                                      for rq in (r.get("reqs") or [])}})
    worker = sorted(driver.split_src(recs, "worker"),
                    key=lambda r: (r.get("seq") is None, r.get("seq") or 0))
    return recs, sched, worker


# ---------------------------------------------------------------------------
# synthetic world: one force3-style boot window (launch_k=4, requested_k=3)
# plus an oscillating group (3<->4) for the lag/transition checks.
# Batch order is deliberately NOT id-sorted (C2 order contract).
#
# Round-4 phase model: a request's verify window == the PREVIOUS proposal's
# width (k_eff, one-step lag) -- there is no launch-K cold-start seeding on
# the sync path, so a first verify is ALSO k_eff-wide. Worker truth: every
# batch row samples k_eff positions (the chunked-prefill row's samples are
# real; only its scheduler-side window stays empty).
# ---------------------------------------------------------------------------
LAUNCH_K = 4
FORCE_K = 3


def worker_spp(bs, k_eff):
    """The speculator's executed-sample counter on this drafter:
    num_reqs samples at every position 0..k_eff-1."""
    return [bs if p < k_eff else 0 for p in range(LAUNCH_K)]


def synth_window():
    log = FakeLogger()
    ev._SEQS.clear()

    # -- step 1: pure decode, 4 rows; r001's FIRST verify keeps its first
    #    proposal's width (3 == force_k; no launch-K cold start) --
    emit_worker(
        log,
        k_eff=FORCE_K,
        launch_k=LAUNCH_K,
        sampled_per_pos=worker_spp(4, FORCE_K),
        req_ids=["r003", "r001", "r004", "r002"],
        idx_rows=[0, 1, 2, 3],
    )
    emit_sched(
        log,
        launch_k=LAUNCH_K,
        requested_k=FORCE_K,
        scheduled_req_ids=["r003", "r001", "r004", "r002"],
        spec_windows={"r003": [-1, -1, -1], "r001": [-1, -1, -1],
                      "r004": [-1, -1, -1], "r002": [-1, -1, -1]},
        acc_rej={"r003": (2, 1), "r001": (0, 3), "r004": (1, 2), "r002": (3, 0)},
        mode="force",
    )

    # -- step 2: mixed batch -- 3 decode rows + 1 chunked-prefill row.
    #    WORKER TRUTH: the prefill row samples too (valid_len == k_eff);
    #    only its scheduler-side window stays empty (absent below). --
    emit_worker(
        log,
        k_eff=FORCE_K,
        launch_k=LAUNCH_K,
        sampled_per_pos=worker_spp(4, FORCE_K),
        req_ids=["r005", "r006", "r003", "r004"],
        idx_rows=[1, 2, 0, 3],
    )
    emit_sched(
        log,
        launch_k=LAUNCH_K,
        requested_k=FORCE_K,
        scheduled_req_ids=["r005", "r006", "r003", "r004"],
        spec_windows={"r005": [-1, -1, -1], "r003": [-1, -1, -1],
                      "r004": [-1, -1, -1]},
        acc_rej={"r005": (2, 1), "r003": (1, 2), "r004": (2, 1)},
        mode="force",
    )

    # -- steps 3-6: oscillation 3<->4 with one-step-lag windows; row remap
    #    (idx_mapping re-derived per step) and budget trims (exempt) --
    # win(t) <= requested_k(t-1) always (== lag, < trim; a request's first
    # verify is ALSO the previous proposal's width -- round-4 k_eff
    # semantics, no cold start).
    osc_rows = [
        # (req_ids, idx_rows, windows_by_req, requested_k)
        # rk=3: r007's first verify keeps its first proposal's width (3).
        (["r002", "r005", "r007"], [2, 0, 1],
         {"r002": [-1, -1, -1], "r005": [-1, -1, -1], "r007": [-1, -1, -1]}, 3),
        # rk=4: r005 trimmed to 2 (< rk(t-1)=3, exempt); others follow the lag.
        (["r007", "r002", "r005"], [1, 2, 0],
         {"r007": [-1, -1, -1], "r002": [-1, -1, -1], "r005": [-1, -1]}, 4),
        # rk=3: r005's pending 4-wide proposal verifies in full (lag == 4).
        (["r005", "r007", "r002"], [0, 2, 1],
         {"r005": [-1, -1, -1, -1], "r007": [-1, -1, -1], "r002": [-1, -1, -1]}, 3),
        # rk=4: all windows <= rk(t-1)=3.
        (["r002", "r005", "r007"], [1, 2, 0],
         {"r002": [-1, -1, -1], "r005": [-1, -1, -1], "r007": [-1, -1, -1]}, 4),
    ]
    for req_ids, idx_rows, windows, rk in osc_rows:
        emit_worker(
            log,
            k_eff=rk,
            launch_k=LAUNCH_K,
            sampled_per_pos=worker_spp(len(req_ids), rk),
            req_ids=req_ids,
            idx_rows=idx_rows,
        )
        acc = {rid: (max(0, len(w) - 1), 1) for rid, w in windows.items()}
        emit_sched(
            log,
            launch_k=LAUNCH_K,
            requested_k=rk,
            scheduled_req_ids=req_ids,
            spec_windows=windows,
            acc_rej=acc,
            mode="armed",
        )
    return log


def main() -> int:
    print(f"gate harness: {GATE_DIR}")
    print(f"emitter helper: {HERE / 'awdbg1b_evidence.py'}")

    log = synth_window()
    recs, sched, worker = parse_log(log)
    check("parse.both_sources", len(sched) == 6 and len(worker) == 6,
          f"sched={len(sched)} worker={len(worker)}")

    # ---- pairing (join_by_seq) ----
    pairs, unpaired_s, unpaired_w = driver.join_by_seq(sched, worker)
    check("join_by_seq.paired", not unpaired_s and not unpaired_w and len(pairs) == 6,
          f"pairs={len(pairs)} unpaired sched={len(unpaired_s)} worker={len(unpaired_w)}")
    seqs = [s.get("seq") for s, _ in pairs]
    check("seq.monotonic", seqs == sorted(seqs) and len(set(seqs)) == len(seqs),
          f"seqs={seqs}")

    # ---- C2 record shape (exact key sets, mirroring the harness MockWorld;
    # parse_evidence annotates each record with 'ts', so the contract keys
    # are asserted modulo that annotation -- and the RAW emitter record is
    # asserted to carry EXACTLY the contract keys. Round 8: the v2 contract
    # adds the proposal-side fields to the worker line (branch/nt/pf +
    # per-row temp/seed/seed_set/props) and the verify/commit-side fields to
    # the sched rows (rej_idx/bonus + emit/phash/ol/nct + the sampling
    # mirror); the OLD-shape builders (no new kwargs) stay parse-compatible
    # and are asserted to omit the new fields. ----
    w0, s0 = pairs[0][1], pairs[0][0]
    raw_w = ev.build_worker_record(
        k_eff=FORCE_K, launch_k=LAUNCH_K,
        sampled_per_pos=[1, 1, 1, 0], req_ids=["x"], idx_rows=[0])
    raw_s = ev.build_sched_record(
        launch_k=LAUNCH_K, requested_k=FORCE_K, scheduled_req_ids=["x"],
        spec_windows={"x": [-1] * FORCE_K}, acc_rej={"x": (1, 2)}, mode="force")
    check("worker.keys.raw.old_shape", set(raw_w) == {"src", "seq", "t", "k_eff",
                                                      "sampled_per_pos", "reqs"},
          f"{sorted(raw_w)}")
    check("sched.keys.raw", set(raw_s) == {"src", "seq", "t", "launch_k",
                                           "requested_k", "bs", "mode", "reqs"},
          f"{sorted(raw_s)}")
    v2_w = ev.build_worker_record(
        k_eff=FORCE_K, launch_k=LAUNCH_K,
        sampled_per_pos=[1, 1, 1, 0], req_ids=["x"], idx_rows=[0],
        branch="gumbel", num_tokens=8, has_prefill=False,
        temps=[0.0], seeds=[0], seeds_set=[False], props=[[1, 2, 3]])
    check("worker.keys.raw.v2", set(v2_w) == {"src", "seq", "t", "k_eff",
                                              "sampled_per_pos", "branch",
                                              "nt", "pf", "reqs"},
          f"{sorted(v2_w)}")
    check("worker.row.keys.v2", set(v2_w["reqs"][0]) == {
        "id", "valid_len", "row", "steps", "temp", "seed", "seed_set",
        "props"}, f"{sorted(v2_w['reqs'][0])}")
    v2_s = ev.build_sched_record(
        launch_k=LAUNCH_K, requested_k=FORCE_K, scheduled_req_ids=["x"],
        spec_windows={"x": [-1] * FORCE_K}, acc_rej={"x": (1, 2)}, mode="force",
        verify={"x": {"rej_idx": 1, "bonus": 42}},
        commit={"x": {"emit": [1, 42], "phash": "0" * 64, "ol": 2, "nct": 9}},
        sampling={"x": {"temp": 0.0, "seed": 1, "seed_set": True}})
    check("sched.row.keys.v2", set(v2_s["reqs"][0]) == {
        "id", "win", "acc", "rej", "rej_idx", "bonus", "emit", "phash",
        "ol", "nct", "temp", "seed", "seed_set"}, f"{sorted(v2_s['reqs'][0])}")
    check("worker.keys", set(w0) - {"ts"} == {"src", "seq", "t", "k_eff",
                                              "sampled_per_pos", "branch",
                                              "nt", "pf", "reqs"},
          f"{sorted(w0)}")
    check("worker.row.keys", set(w0["reqs"][0]) == {
        "id", "valid_len", "row", "steps", "temp", "seed", "seed_set",
        "props"}, f"{sorted(w0['reqs'][0])}")
    check("sched.keys", set(s0) - {"ts", "reqmap"} == {
        "src", "seq", "t", "launch_k", "requested_k", "bs", "mode", "reqs"},
          f"{sorted(s0)}")
    check("sched.row.keys", set(s0["reqs"][0]) == {
        "id", "win", "acc", "rej", "rej_idx", "bonus", "emit", "phash",
        "ol", "nct", "temp", "seed", "seed_set"}, f"{sorted(s0['reqs'][0])}")
    check("sampled_per_pos.length", all(
        len(w.get("sampled_per_pos") or []) == LAUNCH_K for _, w in pairs),
        "len == launch_k on every worker line")
    check("t.unix_us", all(isinstance(r.get("t"), int) and r["t"] > 10**15
                           for r, _ in pairs), "microsecond unix stamps")

    # ---- C1 transport identity: the emitter ships the counter VERBATIM
    # (the same list that feeds the C1 Prometheus counter) ----
    marker = [9, 0, 0, 0]
    marked = ev.build_worker_record(
        k_eff=1, launch_k=LAUNCH_K, sampled_per_pos=marker,
        req_ids=["x"], idx_rows=[0])
    check("c1.counter_shipped_verbatim",
          marked["sampled_per_pos"] == [9, 0, 0, 0],
          "sampled_per_pos == the speculator counter, not a derivation")

    # ---- P1 contract shape (force3 group = the first two steps) ----
    force_pairs = pairs[:2]
    check("P1.A2.requested_k", all(s.get("requested_k") == FORCE_K
                                  for s, _ in force_pairs))
    check("P1.A2b.launch_k", all(s.get("launch_k") == LAUNCH_K
                                 for s, _ in force_pairs))
    check("P1.A6b.k_eff", all(w.get("k_eff") == FORCE_K for _, w in force_pairs))
    check("P1.A3b.no_sample_at_K", all(
        (w.get("sampled_per_pos") or [0] * LAUNCH_K)[FORCE_K] == 0
        for _, w in force_pairs),
        "no sample events at position 3 (the trim discriminator)")
    support = {p for _, w in force_pairs
               for p, v in enumerate(w.get("sampled_per_pos") or []) if v > 0}
    check("P1.A3.sampled_support", support == set(range(FORCE_K)),
          f"support={sorted(support)}")
    # P1.A6 / round-4 worker truth: EVERY row samples (valid_len == k_eff),
    # including the mixed step's chunked-prefill row -- only its
    # scheduler-side window stays empty (win 0 on the sched line).
    check("P1.A6.valid_lengths", all(
        r["valid_len"] == FORCE_K for _, w in force_pairs for r in w["reqs"]),
        "worker truth: every batch row samples k_eff positions")
    step2 = force_pairs[1]
    prefill_rows = [r for r in step2[1]["reqs"]
                    if step2[0]["reqmap"].get(r["id"], {}).get("win", 0) == 0]
    check("worker_truth.prefill_row_samples",
          len(prefill_rows) == 1
          and prefill_rows[0]["valid_len"] == FORCE_K
          and prefill_rows[0]["steps"] == list(range(FORCE_K)),
          "the windowless (chunked-prefill) row consumes its samples; its "
          "SCHED window stays empty")
    check("P1.A5.no_launch_only_samples", all(
        v == 0 for _, w in force_pairs
        for p, v in enumerate(w.get("sampled_per_pos") or []) if p >= FORCE_K),
        "positions >= force_k silent")

    # ---- round-4 A3c uniform executed sampling: sum(spp) == bs x k_eff
    # per line, with bs from the PAIRED sched line (independent source) ----
    uni_bad = []
    for s, w in pairs:
        k_eff = w.get("k_eff")
        bs = s.get("bs") or 0
        spp_sum = sum(w.get("sampled_per_pos") or [])
        if spp_sum != bs * k_eff:
            uni_bad.append((s.get("seq"), bs, k_eff, spp_sum))
    check("P1.A3c.uniform_executed_sampling", not uni_bad,
          f"sum(sampled_per_pos) == bs x k_eff on every paired line"
          + (f"; offenders={uni_bad[:2]}" if uni_bad else ""))

    # ---- P2.B4 aggregate identity per paired step (all 6 steps) ----
    ident_bad = []
    for s, w in pairs:
        spp_sum = sum(w.get("sampled_per_pos") or [])
        vl_sum = sum(r.get("valid_len") or 0 for r in w["reqs"])
        k_eff = w.get("k_eff")
        problems = []
        if spp_sum != vl_sum:
            problems.append(f"sum(spp)={spp_sum} != sum(valid_len)={vl_sum}")
        if any((r.get("valid_len") or 0) not in (0, k_eff) for r in w["reqs"]):
            problems.append("valid_len not in {0, k_eff}")
        if (s.get("bs") or 0) != len(w["reqs"]):
            problems.append(f"sched bs={s.get('bs')} != worker rows={len(w['reqs'])}")
        if problems:
            ident_bad.append((s.get("seq"), problems))
    check("P2.B4.reorder_identity", not ident_bad, f"{ident_bad[:2]}")

    # ---- C2 order contract: reqs in batch order, NOT id-sorted ----
    order = [r["id"] for r in s0["reqs"]]
    check("order.batch_order", order == ["r003", "r001", "r004", "r002"],
          f"{order}")
    check("order.not_id_sorted", order != sorted(order),
          "a canonical emitter would blind B4b order-compare")

    # ---- harness check functions against the synthetic lines ----
    res = driver.PointResult("P1")
    ok_align = driver._alignment_check(worker, res, "P1.A8.alignment_rows",
                                       sched=sched)
    a8 = res.check("P1.A8.alignment_rows")
    check("harness.P1.A8.alignment_rows", ok_align and a8.status == driver.PASS,
          a8.detail[:160] if a8 else "no check")
    check("harness.row_accounting", "row accounting" not in (a8.detail if a8 else ""),
          "worker row count == sched bs on every paired step")

    res2 = driver.PointResult("P2")
    driver._lag_check(sched, res2, "P2.B2.one_step_lag")
    b2 = res2.check("P2.B2.one_step_lag")
    check("harness.P2.B2.one_step_lag", b2.status == driver.PASS,
          b2.detail[:160])

    viol = driver._window_bound_violations(sched, LAUNCH_K)
    check("harness.P1.A4b.window_bounds", not viol, f"{viol[:2]}")

    # ---- round-3 malformed-instrumentation contract (never a crash) ----
    bad = dict(w0)
    del bad["reqs"]
    check("malformed.reqs_omitted",
          driver._reqs_field_problem(bad) == "absent", "reqs omitted -> absent")
    bad2 = dict(w0)
    bad2["reqs"] = 5
    check("malformed.reqs_scalar",
          driver._reqs_field_problem(bad2) is not None, "reqs=5 -> malformed")
    bad3 = dict(w0)
    bad3["reqs"] = [dict(w0["reqs"][0], steps=5)]
    bad3["reqs"] = bad3["reqs"] + list(w0["reqs"][1:])
    res3 = driver.PointResult("P1")
    driver._alignment_check([bad3], res3, "P1.A8.alignment_rows")
    a8b = res3.check("P1.A8.alignment_rows")
    check("malformed.steps_scalar.status",
          a8b is not None
          and a8b.status == driver.MISSING_MALFORMED_INSTRUMENT,
          (a8b.detail[:120] if a8b else "no check"))
    res4 = driver.PointResult("P1")
    driver._alignment_check([bad], res4, "P1.A8.alignment_rows")
    a8c = res4.check("P1.A8.alignment_rows")
    check("malformed.reqs_omitted.status",
          a8c is not None and a8c.status == driver.MISSING_INSTRUMENT,
          (a8c.detail[:120] if a8c else "no check"))

    # ---- width-transition alignment across the oscillation (P2.B6b) ----
    trans_bad = []
    for (s1, w1), (s2, w2) in zip(pairs, pairs[1:]):
        if w1.get("k_eff") == w2.get("k_eff"):
            continue
        for r in w2["reqs"]:
            vl = r.get("valid_len") or 0
            if vl and list(r.get("steps") or []) != list(range(vl)):
                trans_bad.append((s2.get("seq"), r.get("id")))
    check("harness.P2.B6b.transition_alignment", not trans_bad,
          f"k_eff transitions re-cover 0..k_eff-1; offenders={trans_bad[:2]}")

    # ---- round 5: the warmup-offset pairing defect (live gate run
    # 2026-10-14) ---- the runner-side JIT kernel warmup emitted six boot
    # worker lines the scheduler never paired (rows 8/8/2/2/1/1, k_eff ==
    # launch, requested 0), so the worker seq counter ran 6 ahead of the
    # sched seq counter for the whole boot. The synthetic window below
    # replays that shape and proves the harness's pairing/row-accounting
    # checks catch it: the mispaired join flags the row-accounting
    # mismatch and six worker lines stay unpaired. (The round-5 emitter
    # gates on scheduled batches -- requested_k > 0 -- so the live emitter
    # no longer produces these lines; this window is the DEFECT the gate
    # must never silently accept.)
    wlog = FakeLogger()
    ev._SEQS.clear()
    warmup_rows = [8, 8, 2, 2, 1, 1]
    for i, nrows in enumerate(warmup_rows):
        emit_worker(
            wlog,
            k_eff=LAUNCH_K,
            launch_k=LAUNCH_K,
            sampled_per_pos=[nrows if p < LAUNCH_K else 0
                             for p in range(LAUNCH_K)],
            req_ids=[f"_warmup_{j}_" for j in range(nrows)],
            idx_rows=list(range(nrows)),
        )
    # two scheduled steps after the warmup: worker seqs continue at 7..8
    # while the sched counter starts at 1 -- the live desync signature
    for seq_off, win in ((7, 3), (8, 3)):
        emit_worker(
            wlog,
            k_eff=FORCE_K,
            launch_k=LAUNCH_K,
            sampled_per_pos=worker_spp(4, FORCE_K),
            req_ids=["r101", "r102", "r103", "r104"],
            idx_rows=[0, 1, 2, 3],
        )
        emit_sched(
            wlog,
            launch_k=LAUNCH_K,
            requested_k=FORCE_K,
            scheduled_req_ids=["r101", "r102", "r103", "r104"],
            spec_windows={rid: [-1] * win
                          for rid in ("r101", "r102", "r103", "r104")},
            acc_rej={rid: (2, 1)
                     for rid in ("r101", "r102", "r103", "r104")},
            mode="force",
        )
    _recs, wsched, wworker = parse_log(wlog)
    _pairs, _us, _uw = driver.join_by_seq(wsched, wworker)
    check("round5.warmup_unpaired_worker_lines",
          len(_uw) == len(warmup_rows)
          and _uw[0].get("seq") == 3,
          f"unpaired worker lines={len(_uw)} (want {len(warmup_rows)}: the "
          "warmup offsets the seq counters); first offender seq="
          f"{_uw[0].get('seq') if _uw else None}")
    res5 = driver.PointResult("P1")
    ok5 = driver._alignment_check(wworker, res5, "P1.A8.alignment_rows",
                                  sched=wsched)
    a85 = res5.check("P1.A8.alignment_rows")
    check("round5.warmup_mispair_flags_row_accounting",
          not ok5 and a85 is not None and "row accounting" in a85.detail,
          (a85.detail[:160] if a85 else "no check"))

    print()
    if FAILURES:
        print(f"FAIL ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("PASS: synthetic AWDBG1B worker+sched lines parse and satisfy the "
          "round-4/5 P1/P2 contract shape through the harness's own checks "
          "(including the round-5 warmup-offset pairing defect window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
