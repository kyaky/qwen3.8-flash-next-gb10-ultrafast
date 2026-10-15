#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Selftest for the stage-1b gate harness (driver.py).

Validates the harness LOGIC against a mocked world: no server, no docker, no
workload. Every scenario encodes a behavior of the (future) stage-1b boot -
including the exact defect classes the D-009 review demands the gate catch -
and the selftest asserts the harness returns the right verdict for each:

  good paths        -> PASS
  trim defect       (drafter generates wide, scheduler trims)  -> P1 FAIL
  missing counter   (stage-1b patch absent)                    -> MISSING_INSTRUMENT
  missing install   (controller fell back to static K)         -> P1 FAIL
  lag defect        (verify window follows CURRENT width)      -> P2 FAIL
  stale suffix      (samples leak into the launch-only band)   -> P2 FAIL
  row collision     (two requests share a req_state row)       -> P1 FAIL
  column misalign   (per-request draft steps shifted;          -> P2 FAIL
                     aggregate support/identity stay clean)
  trims legitimate  (budget trims during oscillation)          -> P2 PASS
  canonical emitter (id-sorted evidence order blinds order)    -> P2 MISSING_INSTR
  steps scalar      (reqs[].steps not an array: malformed      -> P2 MISSING_INSTR
                     instrumentation, never a run-killing crash)
  reqs scalar       (reqs field not an array: B4 offender)     -> P2 FAIL
  empty reqs        (P1 reqs payload dropped: row accounting)  -> P1 FAIL
  omitted reqs      (P1 reqs field absent: MISSING_INSTRUMENT) -> P1 MISSING_INSTR
  zero width        (worker derives k_eff=0: B3 offender,      -> P2 FAIL
                     0 is not a legal width on this stack)
  pinned drafter    (scheduler oscillates, drafter ignores)    -> P2+P5 FAIL
                     (round 4: the launch-width windows also break
                     B2's lag bound; P5.E2 names the reason)
  spp fabricated    (pre-C1 window-model sampled_per_pos)      -> P1 FAIL via A3c
                     (round 4: the uniform-execution identity;
                     A6 exempts the valid_len==0 rows it produces)
  first-window      (pre-1b launch-width carrier: the first    -> P3 FAIL
  carrier            verify window is launch_k, not the first
                     proposal's width -- inverted discrimination
                     fixed, round 4)
  token divergence  (greedy IDs differ from the MTP=3 baseline)-> P4 FAIL
  distribution drift(acceptance profile moved)                 -> P4 FAIL
  async enabled     (--async-scheduling present)               -> P3 FAIL
  async AUTO-enable (clean argv, sync channel starved:        -> P3 FAIL
                     the round-5 live discovery)
  own-flag pass     (the harness's --no-async-scheduling boot  -> P3 PASS
                     flag must NOT self-trip S7, round 6)
  row trim          (token-budget surplus rows in the worker   -> P1/P2 PASS
                     line, absent from the sched line: the
                     round-6 B4/A8/A3c exemption)
  row drop          (the worker emitter drops a SCHEDULED row  -> P1/P2 FAIL
                     -- the exemption must NOT mask it, round 6)
  warmup-offset     (six JIT-warmup worker records the        -> P1/P2/P5 FAIL
                     scheduler never paired -- the round-5
                     live pairing defect)
  c1 silent         (C1 counter present but its staged        -> P1 FAIL
                     transport never runs -- the round-5
                     async-starved zero delta)
  s6 short probe    (the reuse probe tokenizes below one      -> P3 FAIL
                     1,600-token prefix block)
  evidence silent   (counter present, emitters emit nothing:  -> P1 MISSING_INSTR
                     the round-7 fresh-boot A0 artifact class;
                     the boot-manager instrument probe makes
                     the live equivalent a real emitter fault)
  s4 untuned shape  (the un-calibrated long prompt leaves a    -> P3 WARN
                     huge budget leftover: no trims, the
                     coverage gap is reported, not relaxed)
  round 8 (P2 stage-1 closure, D-014 conds. 1+2):
  c2-fields-absent  (worker lines without the round-8        -> P1 MISSING_INSTR
                     proposal-side fields: an emitter
                     predating the field contract)
  c2-props-malformed(props scalar: wrongly typed,            -> P1 MISSING_INSTR
                     never a crash)
  c2-join-missing   (a dropped worker record leaves a        -> P1 FAIL
                     verify without its producer)
  c2-argmax-mismatch(greedy emit diverges from the           -> P1 FAIL
                     accepted argmax prefix: the
                     target-logit/argmax evidence fires)
  c2-hash-stale     (the committed-prefix hash does not      -> P1 FAIL
                     move while the prefix grows)
  no-awfp           (the fingerprint layer absent: a         -> P1 MISSING_INSTR
                     pre-iter7f image)
  awfp-probe-fail   (the behavioral shim probe FAILED:       -> P1 FAIL
                     the shim absent/unwired/gated off)
  awfp-code-bad     (the marlin code-state check failed:     -> P1 FAIL
                     atomic_add call sites or shim wiring)
  P4 same-shape     (off@K4 vs force4: the committed-prefix  -> P4 PASS
                     chains and the policy hash are
                     byte-identical across the paired boots)
  P4 chain-diverge  (one committed-prefix hash differs:      -> P4 FAIL
                     D5 fires)
  stop-trim proof   (the astra hold of 2026-10-15: a        -> P1 PASS
                     shortened matching prefix passes only
                     on the commit record's stopped=True --
                     the EOS case and the length cap alike)
  stop-trim-unproven(missing stopped / stopped=False / an    -> P1 FAIL
                     empty emit: termination unproven, an
                     explicit failure, never a silent pass)

Also exercises the assert-guarded config-dir builder against the REAL
recipe/config/v18a-g2 serve.sh text (read-only; the repo original is never
touched) - including the round-7 belt-and-braces contract (the committed
config carries the SPECDBG passthrough itself, byte-identical to the
harness's insert, and a stripped clean copy must be re-inserted back to the
committed bytes) - the round-7 point-to-boot plan (P1/P3/P4 on the forced
boot, P2/P5 on the armed-oscillation boot, asserted in gate_boot_plan AND in
every point runner), and the safety refusals (run without --allow-boot,
capture-baseline without --live-ok).

Usage: python3 driver.py selftest [--out DIR]
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import driver as D  # noqa: E402


def _ids(n: int, salt: int) -> list[int]:
    return [(i * 7 + salt) % 150000 for i in range(n)]


def _replay_chains(launch_k: int) -> dict:
    """Replay the capture-baseline chain protocol on a fresh mock boot with
    a constant requested_k == launch_k rhythm (the off floor's shape) and
    harvest the per-prompt committed-prefix chains exactly like
    driver.capture_baseline_outputs does on a live boot."""
    world = D.MockWorld("offk4", launch_k=launch_k, force_k=launch_k)
    server = D.MockServer(world, outputs={})
    logs = D.MockLogs(world.log_lines)
    chains = {}
    for key, prompt in D.P4_PROMPTS:
        t0 = world.clock()
        server.chat([{"role": "user", "content": prompt}],
                    max_tokens=64, temperature=0.0)
        t1 = world.clock()
        recs = D.parse_evidence([x for x in logs.since(t0) if x[0] <= t1])
        fresh = D._first_appearance_chains(D.split_src(recs, "sched"))
        if fresh:
            _, chain = max(fresh.items(), key=lambda kv: (len(kv[1]), kv[0]))
            chains[key] = chain
        else:
            chains[key] = []
    return chains


def baseline_fixture(seeds: list[int], launch_k: int = 3,
                    with_chains: bool = True) -> dict:
    sampled = _ids(24, 7)
    out = {
        "captured_at": 0.0,
        "greedy": {key: _ids(24, salt=i) for i, (key, _) in enumerate(D.P4_PROMPTS)},
        "sampled": {str(s): list(sampled) for s in seeds},
        "accept_profile": {0: 0.7, 1: 0.45, 2: 0.2},
        # round 8: the floor's shape + C2 committed-prefix chains (the D5
        # identity comparison's floor) + the fingerprint policy record
        "launch_k": launch_k,
        "awfp_policy_sha": D._mock_policy_sha(launch_k),
        "awfp_probe_passed": True,
    }
    if with_chains:
        out["c2_chains"] = _replay_chains(launch_k)
    return out


def make_ctx(scenario: str, out_dir: Path, outputs: dict | None = None,
             argv_extra: list[str] | None = None,
             cfg: dict | None = None) -> tuple[D.MockWorld, D.GateContext]:
    cfg = copy.deepcopy(cfg or D.load_config(None))
    world = D.MockWorld(scenario, launch_k=int(cfg["boot"]["launch_k"]),
                        force_k=int(cfg["boot"]["force_k"]))
    world.server = D.MockServer(world, outputs=outputs or {})
    argv = ["vllm", "serve", "/model", "--host", "0.0.0.0"] + (argv_extra or [])
    ctx = D.GateContext(cfg=cfg, metrics=D.MockMetrics(world),
                        logs=D.MockLogs(world.log_lines), server=world.server,
                        out_dir=out_dir, boot_ts=world.t0 - 5.0,
                        launch_k=int(cfg["boot"]["launch_k"]), argv=argv,
                        install_log_ok=None)
    return world, ctx


CASES = [
    # (label, scenario, runner, argv_extra, use_baseline, outputs_mutant,
    #  expect_status, require_pass, require_fail[, cfg_mutator])
    ("P1 good forced-3", "forced3", "P1", [], False, None,
     D.PASS, ["P1.A3.sampled_support", "P1.A3b.no_sample_at_K",
              "P1.A3c.k_eff_matches_request", "P1.A4a.window_counter_reconcile",
              "P1.A4b.window_bounds", "P1.A6b.k_eff",
              "P1.A7.accepted_pos_reconcile", "P1.A8.alignment_rows",
              # round 8: the runtime fingerprints + the C2 field completion
              "P1.FP.code_state", "P1.FP.probe", "P1.FP.failures",
              "P1.C2.fields_worker", "P1.C2.fields_sched",
              "P1.C2.proposal_join", "P1.C2.argmax_greedy",
              "P1.C2.prefix_hash"], []),
    ("P1 trim defect (generate-4-verify-3)", "trim-defect", "P1", [], False, None,
     D.FAIL, [], ["P1.A3b.no_sample_at_K"]),
    ("P1 missing stage-1b counter", "no-counter", "P1", [], False, None,
     D.MISSING_INSTRUMENT, [], []),
    ("P1 controller fell back to static K", "no-install", "P1", [], False, None,
     D.FAIL, [], ["install.logged"]),
    ("P1 alignment defect: two requests share a req_state row",
     "align-row-collision", "P1", [], False, None,
     D.FAIL, [], ["P1.A8.alignment_rows"]),

    ("P1 window-stall: emitter silent across the P1 window (counters advance)",
     "window-stall", "P1", [], False, None,
     D.MISSING_INSTRUMENT, [], ["P1.A1.coverage"]),
    ("P1 unpaired worker line escapes row accounting",
     "unpaired-worker", "P1", [], False, None,
     D.FAIL, [], ["P1.A8b.pairing"]),
    # round 5 (live defect from the 2026-10-14 gate run): the runner-side
    # JIT kernel warmup emitted six boot-time worker lines the scheduler
    # never paired, offsetting the worker seq counter by 6 for the whole
    # boot. The FIXED emitter gates on scheduled batches (requested_k > 0),
    # and these cases prove the harness catches the unfixed shape.
    # P1 note: in the mock world the P1 workload keeps bs constant at 4, so
    # the mispaired lines' row counts and bs x k_eff identities coincide --
    # the offset is caught by the pairing check (A8b), exactly like the live
    # run, where varying batch sizes also tripped A8/A3c.
    ("P1 warmup-offset worker records break the seq pairing",
     "warmup-proposes", "P1", [], False, None,
     D.FAIL, [], ["P1.A8b.pairing"]),
    ("P2 warmup-offset worker records fail the step identities",
     "warmup-proposes", "P2", [], False, None,
     D.FAIL, [], ["P2.B4.reorder_identity"]),
    ("P5 warmup-offset worker records corrupt the armed-drive join",
     "warmup-proposes", "P5", [], False, None,
     D.FAIL, [], ["P5.E2.drive_reaches_drafter"]),
    # round 5 (live defect): the C1 counter was present but its staged
    # transport never ran (the async scheduler starved update_draft_token_ids
    # -- SchedulerConfig.async_scheduling=None auto-resolves to True for MTP
    # boots), so the deltas stayed at zero over 23 steps / 249 evidence
    # samples. A3/A3c must fail on the dead counter.
    ("P1 C1 counter dead (staged transport never runs) fails A3/A3c",
     "c1-silent", "P1", [], False, None,
     D.FAIL, [], ["P1.A3.sampled_support", "P1.A3c.k_eff_matches_request"]),
    # round 4: the pre-C1 emitter fabricated sampled_per_pos/valid_len from
    # the scheduler-visible windows (rows without one contributed nothing).
    # Under worker truth the A3c uniform-execution identity
    # (sum(spp) == bs x k_eff, bs from the PAIRED sched line) catches that
    # fabrication, while A6 -- which exempts valid_len==0 rows exactly like
    # A3c models them -- stays silent on the same lines.
    ("P1 fabricated sampled_per_pos (pre-C1 window model) is caught by A3c",
     "spp-fabricated", "P1", [], False, None,
     D.FAIL, ["P1.A6.valid_lengths"], ["P1.A3c.k_eff_matches_request"]),

    ("P2 good oscillation", "osc-good", "P2", [], False, None,
     D.PASS, ["P2.B1.square_wave", "P2.B2.one_step_lag",
              "P2.B3.samples_follow_current", "P2.B4.reorder_identity",
              "P2.B5.no_stale_suffix", "P2.B6.alignment_rows",
              "P2.B6b.transition_alignment", "P2.B6c.row_remap_observed",
              # round 8: fingerprints + the C2 join across width transitions
              "P2.FP.code_state", "P2.FP.probe", "P2.C2.fields_worker",
              "P2.C2.fields_sched", "P2.C2.proposal_join",
              "P2.C2.argmax_greedy", "P2.C2.prefix_hash"], []),
    ("P2 lag defect (window follows current k)", "osc-lag-defect", "P2", [], False, None,
     D.FAIL, [], ["P2.B2.one_step_lag"]),
    ("P2 stale suffix after shrink", "osc-stale", "P2", [], False, None,
     D.FAIL, [], ["P2.B5.no_stale_suffix"]),
    # round 4: with the worker-truth window model (win == the previous
    # proposal's width = k_eff(t-1)) a pinned drafter's launch-width windows
    # exceed the previous requested width on every shrink step, so B2's
    # lag/bounds invariant fires too; P5.E2 remains the explicit
    # "stage-1b consumption MISSING" discriminator with the named reason
    ("P2 pinned drafter (stage-1b consumption missing) breaks the lag bound",
     "osc-pin", "P2", [], False, None,
     D.FAIL, [], ["P2.B2.one_step_lag"]),
    # round-2: budget trims are legitimate during oscillation (lag exemption)
    ("P2 legitimate trims during oscillation do not false-FAIL the lag check",
     "osc-trim", "P2", [], False, None,
     D.PASS, ["P2.B2.one_step_lag"], []),
    # round 6: a request the token budget trimmed to 0 tokens stays in the
    # worker batch (and samples like every row) while being absent from
    # num_scheduled_tokens -- the sched line's bs and reqs exclude it. The
    # row accounting (B4/A8) and A3c's executed-sample identity must exempt
    # exactly those surplus rows, and still fail a dropped SCHEDULED row.
    ("P1 token-budget trim surplus rows are exempt from row accounting",
     "row-trim", "P1", [], False, None,
     D.PASS, ["P1.A8.alignment_rows", "P1.A3c.k_eff_matches_request"], []),
    ("P2 token-budget trim surplus rows are exempt from row accounting",
     "osc-row-trim", "P2", [], False, None,
     D.PASS, ["P2.B4.reorder_identity", "P2.B6.alignment_rows"], []),
    ("P1 a dropped SCHEDULED row still fails A8 under the trim exemption",
     "row-drop", "P1", [], False, None,
     D.FAIL, [], ["P1.A8.alignment_rows"]),
    ("P2 a dropped SCHEDULED row still fails B4 under the trim exemption",
     "osc-row-drop", "P2", [], False, None,
     D.FAIL, [], ["P2.B4.reorder_identity"]),
    # round-2: per-request draft-step misalignment invisible to the aggregates
    ("P2 draft-step column misalignment (aggregates stay clean)",
     "osc-column-mix", "P2", [], False, None,
     D.FAIL, ["P2.B3.samples_follow_current", "P2.B4.reorder_identity"],
     ["P2.B6.alignment_rows"]),
    # round-2: a canonical (id-sorted) emitter blinds order-compare - caught
    ("P2 canonical emitter violates the batch-order contract",
     "osc-canonical", "P2", [], False, None,
     D.MISSING_INSTRUMENT, [], ["P2.B4c.emitter_order_contract"]),
    # round-2 (relabelled round 4): a joining chunked-prefill row must NOT
    # break the aggregate identity. Under WORKER TRUTH the row SAMPLES like
    # every row (valid_len == k_eff; only its scheduler-side window stays
    # empty), so the identity holds through the join.
    ("P2 chunked-prefill row samples like every row (worker truth); identity holds",
     "osc-prefill", "P2", [], False, None,
     D.PASS, ["P2.B4.reorder_identity", "P2.B6.alignment_rows"], []),

    # round-3 N5: a reqs[].steps value of the wrong JSON type (5 / 3.0
    # instead of an array) used to crash run_p2 with an unhandled TypeError
    # at the steps parse - it must fail the alignment check as MALFORMED
    # instrumentation instead, and every other check must keep working
    ("P2 non-array reqs[].steps is malformed instrumentation (no crash)",
     "osc-steps-scalar", "P2", [], False, None,
     D.MISSING_INSTRUMENT, ["P2.B4.reorder_identity"],
     ["P2.B6.alignment_rows"]),
    # round-3 N5: a reqs field that is not a JSON array is an offender in
    # B4's row accounting (never iterated -> never a crash)
    ("P2 non-array reqs field fails B4 instead of crashing the run",
     "osc-reqs-scalar", "P2", [], False, None,
     D.FAIL, [], ["P2.B4.reorder_identity"]),
    # round-3 N6: P1 must NOT pass vacuously when the emitter drops the reqs
    # payload - the alignment check now cross-checks worker rows vs sched bs
    ("P1 empty reqs payload is caught by the row-accounting cross-check",
     "reqs-empty", "P1", [], False, None,
     D.FAIL, [], ["P1.A8.alignment_rows"]),
    ("P1 omitted reqs field is MISSING_INSTRUMENT (never a vacuous pass)",
     "reqs-omitted", "P1", [], False, None,
     D.MISSING_INSTRUMENT, [], ["P1.A8.alignment_rows"]),
    # round-3 N7 (replaces the tautological 'zero-width support is the empty
    # set' unit case, which only tested set(range(0)) == set()): a k_eff==0
    # worker line is an offender for B3 - 0 is NOT a legal width on this
    # stack (the controller's force contract refuses it and the armed EMA
    # path maps a zero slot to config_k; the donor zero-step parity itself is
    # asserted in the controller CPU gate, section 5g-5)
    ("P2 zero-width worker line (k_eff=0) is an offender for B3",
     "osc-zerowidth", "P2", [], False, None,
     D.FAIL, [], ["P2.B3.samples_follow_current"]),

    ("P3 boundary coverage", "p3", "P3", [], False, None,
     D.PASS, ["P3.S1.first_proposal_coldstart", "P3.S2.chunked_first_window",
              "P3.S3.mixed_steps_seen", "P3.S4.shortened_windows",
              "P3.S4b.trim_within_k_eff", "P3.S5.rejection_observed",
              "P3.S6.prefix_reuse", "P3.S6.successor_coldstart",
              "P3.S7.async_excluded"], []),
    # round 7: the S4 trim model is SHAPE-DRIVEN (the calibrated long prompt
    # leaves a small budget leftover at its chunk; the last decode row behind
    # it takes a partial window). The boundary-coverage case above already
    # proves S4 PASS on the tuned shape; this case pins the untuned shape
    # (s4-flat: huge leftover, no trims) to the WARN coverage-gap report - a
    # detector that PASSed without a trim would be vacuous.
    ("P3 S4 untuned shape reports the coverage gap as WARN",
     "s4-flat", "P3", [], False, None,
     D.WARN, [], ["P3.S4.shortened_windows"]),
    # round 7: the fresh-boot A0 artifact class. The emitters are
    # request-driven, so a boot that has served no request has zero evidence
    # lines even with a healthy stage-1b patch - the 2026-10-14 round-7 gate
    # run false-FAILED P1/P2/P5 exactly there. The boot manager's instrument
    # probe removes the live trigger; the mock scenario models an emitter
    # that is installed but SILENT (counter present, no lines), which A0 must
    # still report as MISSING_INSTRUMENT - a real emitter fault, never a
    # vacuous pass.
    ("P1 evidence emitters silent while the counter exists",
     "no-evidence", "P1", [], False, None,
     D.MISSING_INSTRUMENT, [], []),
    # round 6: every harness boot carries --no-async-scheduling in its env
    # EXTRA (the sync exclusion). The round-5 S7 grep matched ANY token
    # containing "async", so the harness's OWN flag self-tripped the check
    # on every boot (P3 false-FAIL with the negated flag named as an async
    # flag). The negated form must be excluded and the check must PASS with
    # a live sync channel and the flag present.
    ("P3 the harness's own --no-async-scheduling boot flag does not "
     "self-trip S7", "p3", "P3", [D.NO_ASYNC_FLAG], False, None,
     D.PASS, ["P3.S7.async_excluded"], []),
    # round 6: an ENABLE flag (--async-scheduling) must still fail S7 even
    # when the sync channel is live (the argv clause is independent).
    ("P3 --async-scheduling in argv fails S7 even with the sync channel live",
     "p3", "P3", ["--async-scheduling"], False, None,
     D.FAIL, [], ["P3.S7.async_excluded"]),
    ("P3 async scheduling must stay excluded", "async-enabled", "P3",
     ["--async-scheduling"], False, None,
     D.FAIL, [], ["P3.S7.async_excluded"]),
    # round 5: the async scheduler can be live with a CLEAN argv (the
    # SchedulerConfig.async_scheduling=None auto-enable for MTP) -- exactly
    # the state the round-4 gate run reported as "async flags: none". The
    # engine-truth signal (the sync draft-token channel's one-time log line)
    # is absent, so S7 must fail where the argv grep cannot see anything.
    ("P3 async AUTO-ENABLE (clean argv, sync channel starved) caught by S7",
     "async-auto", "P3", [], False, None,
     D.FAIL, [], ["P3.S7.async_excluded"]),
    # round 5: the S6 probe must exceed one 1,600-token prefix-cache block;
    # a probe that tokenizes short is a probe-shape FAIL, never a vacuous
    # pass (the round-4 8-token probe recorded hits delta 0.0 live).
    ("P3 S6 probe below one prefix block fails as a probe-shape defect",
     "s6-short", "P3", [], False, None,
     D.FAIL, [], ["P3.S6.prefix_reuse"]),
    # round 4: the pre-1b launch-width carrier (k_eff pinned at launch while
    # the scheduler asks for 3) must FAIL the k_eff-semantics first-window
    # checks -- the old launch_k expectation passed on the defective engine
    # and failed the correct one (inverted discrimination)
    ("P3 pre-1b launch-width carrier fails the first-window k_eff checks",
     "launch-carrier", "P3", [], False, None,
     D.FAIL, [], ["P3.S1.first_proposal_coldstart"]),

    ("P4 exactness vs baseline", "forced3", "P4", [], True, None,
     D.PASS, ["P4.D1.greedy_token_identity", "P4.D2.seeded_reproducible",
              "P4.D3.cross_config_agreement", "P4.D4.distribution_gate",
              "P4.FP.code_state", "P4.FP.probe"],
     [],
     None,
     # round 8 + R-030b: the legacy MTP=3 floor is shape-mismatched for the
     # K=4 gate boot, so the same-shape-only comparisons report SKIP with
     # the reason stated (a PASS here would be the old PASS-on-skip defect)
     ["P4.FP.cross_boot_policy", "P4.D5.prefix_hash_identity"]),
    ("P4 greedy token divergence", "forced3", "P4", [], True, "diverge",
     D.FAIL, [], ["P4.D1.greedy_token_identity"]),
    ("P4 distribution drift", "dist-drift", "P4", [], True, None,
     D.FAIL, [], ["P4.D4.distribution_gate"]),
    # round 8: the SAME-SHAPE matched control (off@K4 vs force4 -- the
    # revised contract's quantity A at the C2 level). With force_k == 4 the
    # paired boots share the launch shape, so the committed-prefix chains
    # and the cache/kernel-selection policy hash must be IDENTICAL.
    ("P4 same-shape committed-prefix identity (off@K4 vs force4)",
     "forced3", "P4", [], True, None,
     D.PASS, ["P4.D5.prefix_hash_identity", "P4.FP.cross_boot_policy"], [],
     {"boot.force_k": "4"}),
    ("P4 committed-prefix divergence fails D5",
     "forced3", "P4", [], True, "chain-diverge",
     D.FAIL, [], ["P4.D5.prefix_hash_identity"],
     {"boot.force_k": "4"}),
    # round 8 (D-014 condition 2): the C2 field completion defect classes
    ("P1 worker lines without the round-8 proposal fields are "
     "MISSING_INSTRUMENT", "c2-fields-absent", "P1", [], False, None,
     D.MISSING_INSTRUMENT, [], []),
    ("P1 a scalar props value is malformed instrumentation (no crash)",
     "c2-props-malformed", "P1", [], False, None,
     D.MISSING_INSTRUMENT, [], []),
    ("P1 a dropped worker record leaves a verify without its producer",
     "c2-join-missing", "P1", [], False, None,
     D.FAIL, [], ["P1.C2.proposal_join"]),
    ("P1 greedy emit diverging from the argmax prefix fails the C2 evidence",
     "c2-argmax-mismatch", "P1", [], False, None,
     D.FAIL, [], ["P1.C2.argmax_greedy"]),
    ("P1 a committed-prefix hash that does not move on growth fails the chain",
     "c2-hash-stale", "P1", [], False, None,
     D.FAIL, [], ["P1.C2.prefix_hash"]),
    # round 8 (D-014 condition 1): the runtime fingerprint defect classes
    ("P1 a boot without the fingerprint layer is MISSING_INSTRUMENT",
     "no-awfp", "P1", [], False, None,
     D.MISSING_INSTRUMENT, [], []),
    ("P1 a failed behavioral shim probe fails the FP check",
     "awfp-probe-fail", "P1", [], False, None,
     D.FAIL, [], ["P1.FP.probe"]),
    ("P1 a broken marlin code-state fails the FP check",
     "awfp-code-bad", "P1", [], False, None,
     D.FAIL, [], ["P1.FP.code_state"]),

    ("P5 armed drive reaches drafter", "osc-good", "P5", [], False, None,
     D.PASS, ["P5.E1.armed_live", "P5.E2.drive_reaches_drafter",
              "P5.E3.lag_consistency"], []),
    ("P5 pinned drafter (armed decision does NOT reach the drafter)",
     "osc-pin", "P5", [], False, None,
     D.FAIL, [], ["P5.E2.drive_reaches_drafter"]),
    ("P5 lag defect during oscillation", "osc-lag-defect", "P5", [], False, None,
     D.FAIL, [], ["P5.E3.lag_consistency"]),
]


def _outputs_for(cfg_p4: dict, mutant: str | None) -> dict:
    """Mock server outputs: greedy entries keyed by prompt (temperature 0),
    seeded-sampling entries keyed prompt#t<temp>#s<seed>."""
    base = {prompt: _ids(24, salt=i)
            for i, (_key, prompt) in enumerate(D.P4_PROMPTS)}
    sampled = _ids(24, 7)
    temp = cfg_p4["sample_temperature"]
    prompt0 = D.P4_PROMPTS[0][1]
    for seed in cfg_p4["seeds"]:
        base[f"{prompt0}#t{temp}#s{seed}"] = list(sampled)
    if mutant == "diverge":
        # flip one token id in the greedy short-code output
        out = list(base[prompt0])
        out[5] = (out[5] + 1) % 150000
        base[prompt0] = out
    return base


def run_cases(out_dir: Path, results: list[dict]) -> None:
    cfg0 = D.load_config(None)
    baseline = baseline_fixture(list(cfg0["p4"]["seeds"]))
    for case in CASES:
        (label, scenario, point, argv_extra, use_baseline, mutant,
         expect_status, require_pass, require_fail, cfg_mut,
         require_skip) = \
            (list(case) + [None, None])[:11]
        cfg = copy.deepcopy(cfg0)
        case_baseline = baseline
        if cfg_mut:
            for dotted, value in cfg_mut.items():
                section, key = dotted.split(".")
                cfg[section][key] = value
            # a mutated boot shape gets a SAME-SHAPE baseline fixture (the
            # matched-control contract: off@K vs force@K with the same K)
            case_baseline = baseline_fixture(list(cfg["p4"]["seeds"]),
                                             launch_k=int(cfg["boot"]["launch_k"]))
        if point == "P4" and mutant == "chain-diverge":
            case_baseline = copy.deepcopy(case_baseline)
            for key, chain in (case_baseline.get("c2_chains") or {}).items():
                if chain:
                    chain[-1]["phash"] = ("0" * 63 + "1")
                    break
        outputs = _outputs_for(cfg["p4"], mutant) \
            if point == "P4" and mutant != "chain-diverge" else None
        world, ctx = make_ctx(scenario, out_dir, outputs=outputs,
                              argv_extra=argv_extra, cfg=cfg)
        # Round 7: every case runs with the boot name the plan assigns, so
        # the runners' _assert_boot_phase guards are exercised everywhere.
        ctx.boot_name = D.POINT_PHASES[point]
        res = D.PointResult(point)
        if point == "P1":
            D.run_p1(ctx, res)
        elif point == "P2":
            D.run_p2(ctx, res)
        elif point == "P3":
            D.run_p3(ctx, res)
        elif point == "P4":
            D.run_p4(ctx, res, case_baseline if use_baseline else None)
        elif point == "P5":
            D.run_p5(ctx, res)
        ok = True
        notes = []
        if res.status != expect_status:
            ok = False
            notes.append(f"status {res.status} != expected {expect_status}")
        for cid in require_pass:
            c = res.check(cid)
            if c is None:
                ok = False
                notes.append(f"missing check {cid}")
            elif c.status != D.PASS:
                ok = False
                notes.append(f"{cid} = {c.status}: {c.detail[:160]}")
        for cid in require_fail:
            c = res.check(cid)
            if c is None:
                ok = False
                notes.append(f"missing check {cid}")
            elif c.status == D.PASS:
                ok = False
                notes.append(f"{cid} unexpectedly PASSED (defect not caught)")
        for cid in require_skip or []:
            c = res.check(cid)
            if c is None:
                ok = False
                notes.append(f"missing check {cid}")
            elif c.status != D.SKIP:
                ok = False
                notes.append(f"{cid} = {c.status}: expected SKIP "
                             f"({c.detail[:140]})")
        results.append({"case": label, "scenario": scenario, "point": point,
                        "expect": expect_status, "got": res.status,
                        "ok": ok, "notes": notes,
                        "detail": res.to_json()})
        print(f"[{'OK ' if ok else 'FAIL'}] {label}: got {res.status}"
              + ("" if ok else "; " + "; ".join(notes)))


def unit_checks(results: list[dict]) -> None:
    """Small direct checks of the parsing/joining helpers."""
    # malformed evidence line is recorded, not raised
    lines = [(1.0, "INFO x " + D.TAG + " {not json"),
             (2.0, "INFO y " + D.TAG + ' {"src":"sched","seq":1,"requested_k":3}')]
    recs = D.parse_evidence(lines)
    ok = any(r.get("_bad") for r in recs) and \
        any(r.get("requested_k") == 3 for r in recs)
    results.append({"case": "parse_evidence tolerates malformed lines", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] parse_evidence malformed-line handling")

    # join_by_seq pairing
    s = [{"seq": 1}, {"seq": 3}]
    w = [{"seq": 1}, {"seq": 2}]
    pairs, us, uw = D.join_by_seq(s, w)
    ok = len(pairs) == 1 and len(us) == 1 and len(uw) == 1
    results.append({"case": "join_by_seq pairing", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] join_by_seq pairing")

    # round-3 N5/N6 (replaces the tautological 'zero-width support is the
    # empty set' case, which only tested set(range(0)) == set() and proved
    # nothing): the worker reqs field is type-checked WITHOUT iteration
    # before any consumer touches it, so a malformed emitter payload fails
    # its check (MISSING_INSTRUMENT / MISSING_MALFORMED_INSTRUMENT) instead
    # of crashing the gate run
    ok = (D._reqs_field_problem({"reqs": [{"row": 0, "steps": []}]}) is None
          and D._reqs_field_problem({}) == "absent"
          and D._reqs_field_problem({"reqs": None}) == "absent"
          and D._reqs_field_problem({"reqs": 5}) == "not a JSON array (int)"
          and D._reqs_field_problem({"reqs": [5]})
          == "row 0 is not a JSON object (int)")
    results.append({"case": "reqs field type-check never iterates a malformed value",
                    "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] reqs field type-check (round-3 N5)")

    # round 6: the S7 argv filter must see ENABLE flags only -- the
    # harness's own --no-async-scheduling boot flag (env EXTRA append) is a
    # negation and must NOT self-trip the check, while --async-scheduling
    # (any argparse form) and any other async-bearing token still count.
    cases = [
        ([], []),
        ([D.NO_ASYNC_FLAG], []),
        (["--no_async_scheduling"], []),
        (["--No-Async-Scheduling"], []),
        (["--async-scheduling"], ["--async-scheduling"]),
        (["--async-scheduling=True"], ["--async-scheduling=True"]),
        ([D.NO_ASYNC_FLAG, "--async-scheduling"], ["--async-scheduling"]),
        ([D.NO_ASYNC_FLAG, "--enable-async-output"],
         ["--enable-async-output"]),  # unknown async knobs stay fail-noisy
        (["vllm", "serve", "/model", D.NO_ASYNC_FLAG, "--host", "0.0.0.0"], []),
    ]
    ok = all(D.argv_async_flags(a) == want for a, want in cases)
    results.append({"case": "argv_async_flags excludes negated async flags",
                    "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] argv_async_flags excludes negated "
          "async flags (round 6)")

    # ---- round 8: the C2 field-completion helpers ----
    # the worker field type-guard never iterates a malformed reqs payload
    good_w = {"branch": "gumbel", "nt": 7, "pf": False,
              "reqs": [{"temp": 0.0, "seed": 1, "seed_set": True,
                        "props": [1, 2, 3]}]}
    ok = D._worker_c2_problem(good_w) is None
    ok &= D._worker_c2_problem({"nt": 7, "pf": False, "reqs": []}) == "absent:branch"
    ok &= D._worker_c2_problem({"branch": 5, "nt": 7, "pf": False,
                                "reqs": []}) == "malformed:branch is int, not a str"
    ok &= D._worker_c2_problem({"branch": "g", "nt": True, "pf": False,
                                "reqs": []}) == "malformed:nt is bool, not an int"
    ok &= D._worker_c2_problem({"branch": "g", "nt": 7, "pf": 1,
                                "reqs": []}) == "malformed:pf is int, not a bool"
    ok &= D._worker_c2_problem({"branch": "g", "nt": 7, "pf": False,
                                "reqs": [{"temp": 0.0}]}) == "absent:reqs[0].seed"
    ok &= D._worker_c2_problem({"branch": "g", "nt": 7, "pf": False,
                                "reqs": 5}) is None  # reqs shape: not ours
    ok &= D._worker_c2_problem({"branch": "g", "nt": 7, "pf": False,
                                "reqs": [{"temp": 0.0, "seed": 1,
                                          "seed_set": True, "props": 5}]}) \
        == "malformed:reqs[0].props is int, not a JSON array"
    results.append({"case": "worker C2 field type-guard (round 8)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] worker C2 field type-guard (round 8)")

    good_rq = {"rej_idx": 2, "bonus": 5, "emit": [1, 2, 5], "phash": "a" * 64,
               "ol": 3, "nct": 12, "temp": 0.0, "seed": 1, "seed_set": True}
    ok = D._sched_c2_problem(good_rq) is None
    null_ok = dict(good_rq, rej_idx=None, bonus=None, emit=[])
    ok &= D._sched_c2_problem(null_ok) is None
    ok &= D._sched_c2_problem({k: v for k, v in good_rq.items()
                               if k != "phash"}) == "absent:phash"
    ok &= D._sched_c2_problem(dict(good_rq, phash="xyz")) \
        == "malformed:phash is not a 64-hex sha256 digest"
    ok &= D._sched_c2_problem(dict(good_rq, ol=-1)) \
        == "malformed:ol is -1, not a non-negative int"
    ok &= D._sched_c2_problem(dict(good_rq, emit=[True])) \
        == "malformed:emit element bool, not an int"
    results.append({"case": "sched C2 row type-guard (round 8)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] sched C2 row type-guard (round 8)")

    # the proposal join: the producer is the worker step one seq earlier; a
    # missing producer AFTER the series' first line is a defect, the first
    # line's missing producer is the scrape boundary
    sched = [{"seq": 5}, {"seq": 6}, {"seq": 8}]
    worker = [{"seq": 4}, {"seq": 5}]
    joined, n_b, n_m = D._join_proposals(sched, worker)
    ok = (n_b == 0 and n_m == 1 and joined[0][1] is worker[0]
          and joined[1][1] is worker[1] and joined[2][1] is None)
    joined2, n_b2, n_m2 = D._join_proposals([{"seq": 5}], [])
    ok &= n_b2 == 1 and n_m2 == 0 and joined2[0][1] is None
    results.append({"case": "proposal join boundary + missing producer "
                    "(round 8)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] proposal join (round 8)")

    # the cross-boot prefix-chain comparison
    base = [{"ol": 1, "phash": "a" * 64}, {"ol": 5, "phash": "b" * 64}]
    ok = D.compare_prefix_chains(base, [{"ol": 1, "phash": "a" * 64},
                                        {"ol": 6, "phash": "c" * 64}])["match"]
    cmp_bad = D.compare_prefix_chains(base, [{"ol": 1, "phash": "z" * 64}])
    ok &= not cmp_bad["match"] and cmp_bad["common_ol"] == 1
    ok &= not D.compare_prefix_chains(base, [{"ol": 9, "phash": "a" * 64}])["match"]
    results.append({"case": "compare_prefix_chains joins on committed "
                    "output length (round 8)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] compare_prefix_chains (round 8)")

    # the AWFP line parser + failure collector
    lines = [(1.0, "AWFP " + '{"kind":"CODE","ok":true}'),
             (2.0, "AWFP " + '{"kind":"PROBE","passed":true}'),
             (3.0, "AWFP-FAIL probe exploded"),
             (4.0, "INFO unrelated log line")]
    recs = D.parse_awfp(lines)
    ok = ([r.get("kind") for r in recs] == ["CODE", "PROBE"]
          and all(r.get("ts") for r in recs))
    ok &= D.awfp_failures(lines) == ["AWFP-FAIL probe exploded"]
    results.append({"case": "parse_awfp + awfp_failures (round 8)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] parse_awfp (round 8)")

    # the argmax evidence: a probabilistic (temp > 0) verify row is exempt
    res_p = D.PointResult("U")
    series_p = {
        "sched": [{"seq": 2, "reqs": [{"id": "r", "win": 2, "rej_idx": 1,
                                       "bonus": 9, "emit": [4242, 9],
                                       "phash": "a" * 64, "ol": 2, "nct": 5,
                                       "temp": 0.7, "seed": 3,
                                       "seed_set": True}]}],
        "worker": [{"seq": 1, "k_eff": 2, "branch": "gumbel", "nt": 3,
                    "pf": False,
                    "reqs": [{"id": "r", "valid_len": 2, "row": 0,
                              "steps": [0, 1], "temp": 0.7, "seed": 3,
                              "seed_set": True, "props": [7, 8]}]}],
    }
    D._c2_semantics_check(series_p, res_p, "U.C2", 4)
    c = res_p.check("U.C2.argmax_greedy")
    ok = c is not None and c.status == D.PASS \
        and "probabilistic rows exempt" in c.detail
    results.append({"case": "argmax evidence exempts probabilistic rows "
                    "(round 8)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] argmax probabilistic exemption "
          "(round 8)")

    # the stop-trim exemption (first live firing, record run 2026-10-15):
    # a greedy verify whose request TERMINATED inside the accepted prefix --
    # a committed member of it was the EOS/stop token, the engine stopped
    # the output there, and the verify accounting still counts the compared
    # positions (rej_idx=3 while the output list received only the stop
    # token). The committed prefix must still BE the argmax prefix; a
    # committed token that diverges from it stays an offender.
    #
    # The astra hold of 2026-10-15: a shortened prefix is accepted ONLY on
    # PROOF of termination -- the engine-side commit record's stopped flag
    # (the scheduler's own _update_request_with_output verdict, True for
    # the EOS stop and the max_tokens length cap alike, each with the same
    # trim). A MISSING flag, a False one, or an empty emit without proof is
    # an explicit failure with its cause, never a silent pass.
    def _stop_trim_series(emit, **rq_extra):
        return {
            "sched": [{"seq": 2, "reqs": [dict(
                {"id": "r", "win": 4, "rej_idx": 3, "bonus": 999,
                 "emit": emit, "phash": "a" * 64, "ol": 2, "nct": 5,
                 "temp": 0.0, "seed": 3, "seed_set": True}, **rq_extra)]}],
            "worker": [{"seq": 1, "k_eff": 4, "branch": "greedy", "nt": 3,
                        "pf": False,
                        "reqs": [{"id": "r", "valid_len": 4, "row": 0,
                                  "steps": [0, 1, 2, 3], "temp": 0.0,
                                  "seed": 3, "seed_set": True,
                                  "props": [248046, 198, 248044, 248045]}]}],
        }

    # the pre-existing EOS case: the stop token is the first committed
    # member and the record PROVES the stop (stopped=True) -> PASS
    res_eos = D.PointResult("U")
    D._c2_semantics_check(_stop_trim_series([248046], stopped=True),
                          res_eos, "U.C2", 4)
    c = res_eos.check("U.C2.argmax_greedy")
    ok = c is not None and c.status == D.PASS and "stop-trimmed 1" in c.detail
    res_eos_bad = D.PointResult("U")
    D._c2_semantics_check(_stop_trim_series([248046, 7], stopped=True),
                          res_eos_bad, "U.C2", 4)
    cb = res_eos_bad.check("U.C2.argmax_greedy")
    ok &= cb is not None and cb.status == D.FAIL \
        and "stop-trimmed emit" in cb.detail
    results.append({"case": "argmax stop-trim exemption (EOS inside the "
                    "accepted prefix, stopped=True) + committed-prefix "
                    "divergence still fails", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] argmax stop-trim exemption")

    # a shortened prefix WITH the proof passes at any trim point (the
    # counter is per verify row: this trim commits two of four positions)
    res_proof = D.PointResult("U")
    D._c2_semantics_check(_stop_trim_series([248046, 198], stopped=True),
                          res_proof, "U.C2", 4)
    c = res_proof.check("U.C2.argmax_greedy")
    ok = c is not None and c.status == D.PASS and "stop-trimmed 1" in c.detail
    results.append({"case": "stop-trim accepts a shortened prefix only "
                    "with stopped=True (astra hold 2026-10-15)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] stop-trim stopped=True acceptance")

    # WITHOUT the proof it is a failure with its cause -- a missing flag
    # (an emitter predating the field) and an explicit stopped=False alike
    res_unproven = D.PointResult("U")
    D._c2_semantics_check(_stop_trim_series([248046]), res_unproven,
                          "U.C2", 4)
    c = res_unproven.check("U.C2.argmax_greedy")
    ok = c is not None and c.status == D.FAIL \
        and "no stopped flag" in c.detail \
        and "termination unproven" in c.detail
    res_false = D.PointResult("U")
    D._c2_semantics_check(_stop_trim_series([248046], stopped=False),
                          res_false, "U.C2", 4)
    c = res_false.check("U.C2.argmax_greedy")
    ok &= c is not None and c.status == D.FAIL \
        and "stopped=False" in c.detail
    results.append({"case": "stop-trim without proof fails (missing flag "
                    "and stopped=False), never a silent pass "
                    "(astra hold 2026-10-15)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] stop-trim unproven rejection")

    # the empty emit is the extreme trim: without the proof it must fail
    # too (it would erase the whole window's evidence)
    res_empty = D.PointResult("U")
    D._c2_semantics_check(_stop_trim_series([]), res_empty, "U.C2", 4)
    c = res_empty.check("U.C2.argmax_greedy")
    ok = c is not None and c.status == D.FAIL \
        and "stop-trimmed emit []" in c.detail \
        and "no stopped flag" in c.detail
    results.append({"case": "stop-trim: an empty emit without proof fails "
                    "with its cause (astra hold 2026-10-15)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] stop-trim empty-emit rejection")

    # the length cap trims the same way (the engine's
    # _update_request_with_output sets stopped=True for max_tokens too),
    # labeled by the optional stop/finish marker -- same proof, counted
    # apart in the report
    res_len = D.PointResult("U")
    D._c2_semantics_check(
        _stop_trim_series([248046, 198, 248044], stopped=True,
                          stop_reason="length"), res_len, "U.C2", 4)
    c = res_len.check("U.C2.argmax_greedy")
    ok = c is not None and c.status == D.PASS \
        and "stop-trimmed 1" in c.detail \
        and "length-capped 1" in c.detail
    results.append({"case": "stop-trim: the max_tokens length cap passes "
                    "on stopped=True and is counted apart "
                    "(astra hold 2026-10-15)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] stop-trim length-cap case")


def boot_wiring_checks(cfg: dict, results: list[dict]) -> None:
    """Round 6: the boot's copy source lives on the BootSpec, and the
    post-boot floor validation rejects a poisoned floor before the P4
    capture."""
    spec_baseline, spec_force, spec_osc = D.make_boot_specs(cfg)
    ov_off = spec_baseline.env_overrides()
    ok = (spec_baseline.base_config_rel == cfg["baseline_config_dir"]
          and spec_force.base_config_rel == cfg["base_config_dir"]
          and spec_osc.base_config_rel == cfg["base_config_dir"]
          and ov_off.get("ADAPTIVE_WIDTH_MODE") == "off"
          and ov_off.get("ADAPTIVE_WIDTH_FORCE") == ""
          and spec_force.env_overrides().get("ADAPTIVE_WIDTH_MODE") == "force"
          and spec_osc.env_overrides().get("ADAPTIVE_WIDTH_MODE") == "armed")
    results.append({"case": "BootSpec.base_config_rel wiring (baseline -> "
                    "v16b, gate boots -> base)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] BootSpec.base_config_rel wiring "
          "(round 6)")

    # Round 7: the EXPLICIT point-to-boot plan. P4's fixed-K premise
    # (byte-identical greedy token IDs vs the off baseline) requires the
    # FORCED boot; P2/P5's subject IS the armed oscillation; coverage must be
    # exact and exclusive. The plan is data (POINT_PHASES + gate_boot_plan),
    # run_gate walks it, and every point runner re-asserts the boot it was
    # handed - a wrong assignment can no longer be silent.
    plan = D.gate_boot_plan(cfg)
    assign = {}
    for name, spec, pts in plan:
        for p in pts:
            assign[p] = (name, spec)
    ok = (assign["P4"][0] == "force3" and assign["P1"][0] == "force3"
          and assign["P3"][0] == "force3" and assign["P2"][0] == "osc"
          and assign["P5"][0] == "osc"
          and assign["P4"][1].aw_mode == "force"
          and assign["P4"][1].aw_force == str(int(cfg["boot"]["force_k"]))
          and assign["P2"][1].aw_mode == "armed"
          and assign["P2"][1].aw_force == "oscillate"
          and set(assign) == set(D.POINT_PHASES))
    results.append({"case": "gate_boot_plan assigns P1/P3/P4 -> force3, "
                    "P2/P5 -> osc (round 7)", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] gate_boot_plan assignment (round 7)")

    class _WrongBoot:
        boot_name = "osc"

    class _RightBoot:
        boot_name = "force3"

    raised = False
    try:
        D._assert_boot_phase(_WrongBoot(), "P1")
    except AssertionError:
        raised = True
    ok = raised and D._assert_boot_phase(_RightBoot(), "P1") is None
    results.append({"case": "_assert_boot_phase refuses a wrong-boot point",
                    "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] _assert_boot_phase wrong-boot guard "
          "(round 7)")

    good_argv = ["vllm", "serve", "/model", D.NO_ASYNC_FLAG]
    vcases = [
        ({"MTP": "3"}, good_argv, "off/MTP=3/sync floor", True),
        ({"MTP": "3"}, ["vllm", "serve", "/model"],
         "floor without the sync flag rejected", False),
        ({"MTP": "3", "VLLM_ADAPTIVE_WIDTH_MODE": "armed"}, good_argv,
         "armed floor rejected", False),
        ({"MTP": "4"}, good_argv, "MTP=4 floor rejected", False),
    ]
    ok = True
    for env, use_argv, label, want in vcases:
        got_ok, why = D.validate_booted_baseline(cfg, env=dict(env),
                                                 argv=list(use_argv))
        if got_ok != want:
            ok = False
            print(f"    validate_booted_baseline {label}: got ok={got_ok}"
                  f" ({why})")
    results.append({"case": "validate_booted_baseline floor checks", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] validate_booted_baseline floor "
          "checks (round 6)")


def baseline_boot_checks(cfg: dict, results: list[dict]) -> None:
    """is_valid_baseline_boot reads Config.Env (docker inspect), not argv
    (round-2 attack: the running container carries VLLM_ADAPTIVE_WIDTH_MODE=
    shadow in Config.Env while argv is clean, and the old argv-only grep
    validated that shadow boot as the off floor)."""
    argv = ["vllm", "serve", "/model", "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":3}']
    argv_no_spec = ["vllm", "serve", "/model", "--host", "0.0.0.0"]
    cases = [
        ({"MTP": "3"}, argv, "clean off floor", True),
        ({"MTP": "3", "VLLM_ADAPTIVE_WIDTH_MODE": "off"}, argv,
         "explicit off", True),
        ({"MTP": "3", "VLLM_ADAPTIVE_WIDTH_MODE": "shadow"}, argv,
         "shadow in Config.Env rejected (the round-1 attack)", False),
        ({"MTP": "3", "VLLM_ADAPTIVE_WIDTH_MODE": "armed"}, argv,
         "armed in Config.Env rejected", False),
        ({"MTP": "3", "VLLM_ADAPTIVE_WIDTH_MODE": "force"}, argv,
         "force in Config.Env rejected", False),
        ({"MTP": "4"}, argv, "MTP=4 rejected", False),
        ({}, argv_no_spec, "no MTP in env or argv rejected", False),
    ]
    for env, use_argv, label, want_ok in cases:
        ok, why = D.is_valid_baseline_boot(cfg, env=dict(env), argv=list(use_argv))
        passed = ok == want_ok
        results.append({"case": f"baseline-boot env check: {label}",
                        "ok": passed, "why": why})
        print(f"[{'OK ' if passed else 'FAIL'}] baseline-boot env check: {label}"
              + ("" if passed else f" (got ok={ok}, {why})"))
    # argv-inlined env (launchers that bypass -e): explicit off passes,
    # anything else does not
    ok, _ = D.is_valid_baseline_boot(cfg, env={},
                                     argv=argv + ["-e", "VLLM_ADAPTIVE_WIDTH_MODE=off"])
    results.append({"case": "baseline-boot argv-inlined MODE=off passes", "ok": ok})
    print(f"[{'OK ' if ok else 'FAIL'}] baseline-boot argv-inlined MODE=off passes")
    ok2, why2 = D.is_valid_baseline_boot(
        cfg, env={}, argv=argv + ["-e", "VLLM_ADAPTIVE_WIDTH_MODE=shadow"])
    results.append({"case": "baseline-boot argv-inlined MODE=shadow rejected",
                    "ok": not ok2, "why": why2})
    print(f"[{'OK ' if not ok2 else 'FAIL'}] baseline-boot argv-inlined "
          "MODE=shadow rejected")


def config_dir_checks(cfg: dict, results: list[dict]) -> None:
    """Exercise the assert-guarded config builder on the REAL v18a-g2 files
    (copied into a temp dir; the repository original is never modified)."""
    repo = Path(cfg["repo_root"])
    base_rel = cfg["base_config_dir"]
    if not (repo / base_rel / "serve.sh").exists():
        results.append({"case": "config-dir builder (base config missing; skipped)",
                        "ok": True, "skipped": True})
        print("[SKIP] config-dir builder: base config dir not found")
        return
    tmp = Path(tempfile.mkdtemp(prefix="gate-selftest-"))
    try:
        spec = D.BootSpec("force3", mtp=4, aw_mode="force", aw_force="3",
                          launch_k=4, candidates=[3, 4],
                          base_config_rel=base_rel)
        dst = D.prepare_config_dir(spec, tmp, repo)
        env = (dst / "env").read_text()
        checks = {
            "env MTP=4": "MTP='4'" in env,
            "env mode=force": "ADAPTIVE_WIDTH_MODE='force'" in env,
            "env force=3": "ADAPTIVE_WIDTH_FORCE='3'" in env,
            "env evidence=1": "SPECDBG_EVIDENCE='1'" in env,
            "env has CONFIG_FILE": "ADAPTIVE_WIDTH_CONFIG_FILE=" in env,
            "env lacks host CONFIG path": not re.search(
                r"^ADAPTIVE_WIDTH_CONFIG=", env, re.M),
        }
        aw = json.loads((dst / "aw-gate-config.json").read_text())
        checks["aw candidates top"] = aw.get("candidate_steps") == [3, 4]
        checks["aw candidates tier"] = (aw.get("tiers", {}).get("1", {})
                                        .get("candidate_steps") == [3, 4])
        serve = (dst / "serve.sh").read_text()
        checks["serve.sh AWCFG_ARGS"] = "AWCFG_ARGS" in serve
        checks["serve.sh SPECDBG_ARGS"] = "SPECDBG_ARGS" in serve
        checks["serve.sh mounts config"] = "/aw-gate-config.json" in serve
        # round 5: the sync-scheduling flag rides the env EXTRA (the vllm
        # argv passthrough both v16b and v18a serve.sh expand)
        checks["env EXTRA has --no-async-scheduling"] = \
            D.NO_ASYNC_FLAG in env
        # launch.sh must still be the untouched original (harness edits only
        # env + serve.sh in the copy)
        orig_launch = (repo / base_rel / "launch.sh").read_text()
        checks["launch.sh untouched"] = (dst / "launch.sh").read_text() == orig_launch
        # idempotent re-patch
        checks["serve.sh idempotent"] = \
            D._patch_serve_sh(dst / "serve.sh") == ["serve.sh:already-patched"]
        # idempotent EXTRA append (already present -> no rewrite)
        checks["EXTRA append idempotent"] = \
            D._append_extra_arg(dst / "env", D.NO_ASYNC_FLAG) \
            == f"EXTRA:{D.NO_ASYNC_FLAG}:already-present"
        # guard: candidates above launch_k must be refused
        refused = False
        try:
            D._write_aw_config(dst / "bad.json", [3, 7], 4)
        except AssertionError:
            refused = True
        checks["candidates>launch_k refused"] = refused
        # round 7: the belt-and-braces contract. The COMMITTED v18a-g2
        # serve.sh carries the gate plumbing itself - byte-identical to the
        # harness's insert (one occurrence of each insert text) - so a boot
        # is instrumented even if the insert were skipped; the committed env
        # leaves the SPECDBG_EVIDENCE knob to the harness (set per boot,
        # never default-on in the recipe config).
        repo_serve_text = (repo / base_rel / "serve.sh").read_text()
        checks["committed serve.sh carries the insert verbatim"] = (
            repo_serve_text.count(D._SERVE_INSERT) == 1
            and repo_serve_text.count(D._SERVE_RUN_INSERT) == 1
            and "SPECDBG_ARGS" in repo_serve_text
            and 'VLLM_SPECDBG_EVIDENCE=${SPECDBG_EVIDENCE}' in repo_serve_text)
        checks["committed env leaves SPECDBG_EVIDENCE to the harness"] = (
            "SPECDBG_EVIDENCE" not in (repo / base_rel / "env").read_text())
        ok = all(checks.values())
        results.append({"case": "config-dir builder on real v18a-g2",
                        "ok": ok, "checks": checks})
        print(f"[{'OK ' if ok else 'FAIL'}] config-dir builder on real v18a-g2")
        if not ok:
            print("   failed: " + str([k for k, v in checks.items() if not v]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # round 7: the insert itself must still work from a CLEAN (unpatched)
    # config shape - strip the committed gate block from a copied config dir
    # and prove prepare_config_dir re-inserts it back to the committed bytes
    # (assert-guarded; the second run is a no-op). This is the "insert on a
    # clean copy / idempotent re-run / passthrough present" triple.
    tmp2 = Path(tempfile.mkdtemp(prefix="gate-selftest-clean-"))
    try:
        clean_root = tmp2 / "repo"
        dst_dir = clean_root / base_rel
        shutil.copytree(repo / base_rel, dst_dir)
        serve = (dst_dir / "serve.sh").read_text()
        n_ins = serve.count(D._SERVE_INSERT)
        n_run = serve.count(D._SERVE_RUN_INSERT)
        assert n_ins == 1 and n_run == 1, \
            f"committed insert text not found verbatim ({n_ins}, {n_run})"
        serve = serve.replace(D._SERVE_INSERT, "") \
                     .replace(D._SERVE_RUN_INSERT, "")
        (dst_dir / "serve.sh").write_text(serve)
        spec_clean = D.BootSpec("force3", mtp=4, aw_mode="force", aw_force="3",
                                launch_k=4, candidates=[3, 4],
                                base_config_rel=base_rel)
        dst2 = D.prepare_config_dir(spec_clean, clean_root, clean_root)
        checks2 = {
            "clean-copy insert restores the committed serve.sh bytes":
                (dst2 / "serve.sh").read_text() == repo_serve_text,
            "clean-copy env gets the evidence line":
                "SPECDBG_EVIDENCE='1'" in (dst2 / "env").read_text(),
            "clean-copy insert idempotent on re-run":
                D._patch_serve_sh(dst2 / "serve.sh")
                == ["serve.sh:already-patched"],
        }
        ok2 = all(checks2.values())
        results.append({"case": "config-dir insert on a clean copy (round 7)",
                        "ok": ok2, "checks": checks2})
        print(f"[{'OK ' if ok2 else 'FAIL'}] config-dir insert on a clean "
              "copy (round 7)")
        if not ok2:
            print("   failed: " + str([k for k, v in checks2.items() if not v]))
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
    # round 5: the baseline boot builder against the REAL v16b dir -- the
    # off-floor copy predates the AW serve.sh anchor, so it must be prepared
    # with patch_serve=False (explicitly logged, never silently skipped) and
    # still receive the MTP=3 / sync-scheduling env rewrites.
    base_rel = cfg["baseline_config_dir"]
    if not (repo / base_rel / "serve.sh").exists():
        results.append({"case": "baseline builder (v16b missing; skipped)",
                        "ok": True, "skipped": True})
        print("[SKIP] baseline builder: v16b config dir not found")
        return
    tmp = Path(tempfile.mkdtemp(prefix="gate-selftest-base-"))
    try:
        spec = D.BootSpec("baseline", mtp=3, aw_mode=None, aw_force=None,
                          launch_k=3, candidates=[3], sync_scheduling=True,
                          patch_serve=False, base_config_rel=base_rel)
        dst = D.prepare_config_dir(spec, tmp, repo)
        env = (dst / "env").read_text()
        prep = (dst / "prepare.log").read_text()
        checks = {
            "baseline env MTP=3": "MTP='3'" in env,
            "baseline env EXTRA has --no-async-scheduling":
                D.NO_ASYNC_FLAG in env,
            # round 6: the off floor is pinned EXPLICITLY (mode=off, force
            # knob empty) so an inherited armed/oscillate line can never
            # leak through whatever dir the spec copies.
            "baseline env pins ADAPTIVE_WIDTH_MODE='off'":
                "ADAPTIVE_WIDTH_MODE='off'" in env,
            "baseline env force knob empty":
                "ADAPTIVE_WIDTH_FORCE=''" in env,
            # round 6: nothing mounts aw-gate-config.json on an unpatched
            # serve.sh, so the dead env line must be skipped (logged).
            "baseline env skips the dead CONFIG_FILE line":
                "ADAPTIVE_WIDTH_CONFIG_FILE" not in env,
            "baseline prepare.log names the config source":
                f"base_config_dir:{base_rel}" in prep,
            "baseline serve.sh untouched by the AW insert":
                "AWCFG_ARGS" not in (dst / "serve.sh").read_text(),
            "baseline prepare.log names the serve.sh skip explicitly":
                "serve.sh:skipped" in prep,
        }
        ok = all(checks.values())
        results.append({"case": "baseline builder on real v16b (round 5)",
                        "ok": ok, "checks": checks})
        print(f"[{'OK ' if ok else 'FAIL'}] baseline builder on real v16b")
        if not ok:
            print("   failed: " + str([k for k, v in checks.items() if not v]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def safety_checks(cfg: dict, results: list[dict]) -> None:
    """The harness must refuse to touch the server without explicit flags."""
    cfg = copy.deepcopy(cfg)
    cfg["allow_boot"] = False
    rc = D.run_gate(cfg, {"P1"}, Path(tempfile.mkdtemp(prefix="gate-refuse-")))
    ok = rc == 2
    results.append({"case": "run refuses without --allow-boot", "ok": ok,
                    "rc": rc})
    print(f"[{'OK ' if ok else 'FAIL'}] run refuses without --allow-boot (rc={rc})")

    cfg2 = copy.deepcopy(cfg)
    cfg2["live_ok"] = False
    rc2 = D.capture_baseline(cfg2)
    ok2 = rc2 == 2
    results.append({"case": "capture-baseline refuses without --live-ok",
                    "ok": ok2, "rc": rc2})
    print(f"[{'OK ' if ok2 else 'FAIL'}] capture-baseline refuses without "
          f"--live-ok (rc={rc2})")


def run(out_dir: Path) -> int:
    print("=== gate-stage1b selftest (mocked world; no server, no docker) ===")
    results: list[dict] = []
    unit_checks(results)
    cfg = D.load_config(None)
    boot_wiring_checks(cfg, results)
    config_dir_checks(cfg, results)
    baseline_boot_checks(cfg, results)
    run_cases(out_dir, results)
    safety_checks(cfg, results)
    failed = [r for r in results if not r.get("ok")]
    report = {"selftest": "gate-stage1b", "cases": len(results),
              "failed": len(failed), "results": results}
    path = out_dir / "selftest-report.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nselftest: {len(results) - len(failed)}/{len(results)} OK"
          f" -> {path}")
    if failed:
        for r in failed:
            print(f"  FAILED: {r['case']}: {r.get('notes', '')}")
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else HERE / "selftest-out"
    out.mkdir(parents=True, exist_ok=True)
    sys.exit(run(out))
