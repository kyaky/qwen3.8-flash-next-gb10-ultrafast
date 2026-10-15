---
author: glm-5.3-flash
date: 2026-10-14
task: PLAN.md#P2 (stage-1b gate; implements D-009 verdict section d, codex task-mv6hbxkt)
type: design
status: draft
reviewed_by: pending-codex (round 4)
evidence: agents/glm-5.3-flash-integrator/scripts/gate-stage1b/ (this harness, selftest 66/66 after round 7;
  round 6 was 61/61, round 5 was 52/52, round 4 was 45/45, round 3 was 43/43, round 2 was 35/35,
  round 1 was 22/22);
  round-7 revision basis: the second LIVE gate run
  (runs/2026-10-14-p2-stage1b-gate/gate-run2.log + gate-stage1b-20261014-040427.json;
  core plumbing P2 B1/B2/B3/B4b/B4c/B5/B6/B6b and nearly all of P3 green incl. the new S3
  shape and the >1600-block S6 probe) with three remaining defects - (1) P1/P2/P5
  MISSING_INSTRUMENT: the round-5 emitter gate made the C2 emitters REQUEST-DRIVEN, so the
  A0 "lines since boot" check ran before any request existed on a fresh boot and false-FAILED
  (fix: the boot-manager instrument probe; the belt-and-braces committed v18a-g2 passthrough
  rides along), (2) the P4 point-to-boot assignment made explicit + assert-guarded (run2's
  P4 D1/D2/D3 failures are engine-side findings, not an assignment defect - the round-6 code
  already ran P4 on force3), (3) P3.S4 retuned with a runtime-calibrated prompt plus the
  scheduler-source trigger analysis (see "S4 trigger analysis" below);
  round-6 revision basis: the round-5 re-review (tasks/wrk832rhd.output rereview: fixes-requires-more -
  the two engine root-cause fixes confirmed correct; two BLOCKING round-5 harness defects landed:
  S7's argv grep self-trips on the harness's own --no-async-scheduling flag, and the auto-baseline
  hardcoded base_config_dir (armed v18a-g2) as the off/MTP=3 floor copy source with no post-boot
  re-validation; plus the B4/A8 token-budget trim row-accounting exemption);
  D-009 verdict log state/anchor-excerpts-1b4cd9e363c61925/jobs/task-mv6hbxkt-g88c99.log;
  round-2 attack list agents/glm-5.3-flash-integrator/runs/2026-10-12-p0/2026-10-13-d009-revision.json;
  round-4 revision basis: the adversarial review of the stage-1b implementation
  (commits 8c735b7 + 09196bb), verdict implementation-requires-fixes, 7 attacks landed;
  round-5 revision basis: the first LIVE gate run + the codex alignment
  (runs/2026-10-14-p2-stage1b-gate/gate-run.log + gate-stage1b-20261014-014411.json;
  the six-excess-records / zero-C1-delta alignment review), root causes: the JIT
  kernel warmup's six non-scheduled proposes (gpu/warmup.py warmup_kernels,
  SchedulerOutput.make_empty default num_spec_tokens_to_schedule=0) and the
  SchedulerConfig.async_scheduling=None auto-enable for MTP boots (the AsyncScheduler
  starved update_draft_token_ids; P3.S7's argv-only grep could not see it);
  alignment contract sources runs/2026-10-12-p0/anchor-excerpts-v2/ (rej-utils strides,
  req_state_idx row ownership, per-token draft_step_idx bounds);
  live /metrics scrape 2026-10-13 (counter labels verified on qwen38-flash-dgx:iter7a-20261013)
---

# Stage-1b gate harness (gate-stage1b)

Runnable five-point gate for the P2 stage-1b work (per-call draft width in the
V2 speculator runner). The codex design review D-009 blocked implementation
until exactly this gate existed; every assertion below traces to its section d.

**Role split (hard boundary):** this harness is CPU-side only. It drives HTTP
workloads, scrapes `/metrics`, parses evidence lines, boots/restores configs.
The runner-side instrumentation it asserts against, the debug counter and the
`AWDBG1B` evidence emitters, is stage-1b proper and is SPECIFIED here
(section "Debug-counter contract"), not implemented here.

## Layout

```
scripts/gate-stage1b/
├── README.md           this file: assertions + debug-counter contract
├── driver.py           the harness (stdlib only): sources, boot manager,
│                       P1-P5 runners, orchestrator, CLI
├── gate_selftest.py    mocked-world scenarios proving the harness catches
│                       every D-009 defect class (selftest 66/66 after
│                       round 7; round 6 was 61/61, round 5 was 52/52,
│                       round 4 was 45/45, round 3 was 43/43,
│                       round 2 was 35/35, round 1 was 22/22)
└── gate_config.json    boot specs, workload shape, tolerances
```

Subcommands (default = selftest, deliberately the safe one):

| command | touches server? | touches containers? | purpose |
|---|---|---|---|
| `selftest` | no | no | validate harness logic against the mocked world |
| `check-live` | one read-only GET `/metrics` + `docker inspect` | no | counter inventory; reports whether the stage-1b counter exists |
| `capture-baseline` | small fixed prompt set | no | record P4 baseline token IDs (requires `--live-ok`; refuses unless the running boot is MTP=3/off) |
| `run --points ...` | workloads + boots | **YES - replaces the running container** | execute gate points (requires `--allow-boot`; restores the v16b baseline afterwards) |

Verified on 2026-10-13 against the running server (iter7a-20261013, MTP=3):
`check-live` showed the four standard counters with labels
`{engine, model_name}` (+`{position}` for the per-pos counter) and the
debug counter ABSENT (the correct MISSING_INSTRUMENT condition for a
pre-stage-1b image); argv has no async flags; the running boot is MTP=3
with `VLLM_ADAPTIVE_WIDTH_MODE=shadow` in Config.Env, a valid P4 baseline
numerically (shadow drives no K and at MTP=3 every ladder clamps to
static), and check-live reports that mode explicitly instead of validating
it as the off floor (round-2 fix: the baseline-boot check reads
Config.Env, not argv).

Round 4 (2026-10-14) revised the harness against the adversarial review of
the stage-1b implementation: the mocked world now emits WORKER TRUTH (see
the phase model), the cold-start expectations are k_eff-semantics, A3c is
a uniform-executed-sampling identity, and A6/A3c exempt rows without draft
positions.

Round 5 (2026-10-14, after the first LIVE gate run
`runs/2026-10-14-p2-stage1b-gate/`) fixes what that run exposed:

1. **The six excess worker records were the runner-side JIT kernel warmup.**
   `gpu/warmup.py warmup_kernels` runs 6 real non-dummy steps at worker init
   (prefill + 5 decode steps, batch rows 8/8/2/2/1/1) through fabricated
   `SchedulerOutput.make_empty()` outputs whose `num_spec_tokens_to_schedule`
   stays at the dataclass default **0**. The round-4 worker emitter keyed on
   `is not None` fired on all six (k_eff falls back to the launch width) and
   the scheduler never paired them, so the worker seq counter ran 6 ahead of
   the sched seq counter for the whole boot, exactly the live signature
   (worker head 9 four-width entries vs sched 3; B4's offenders carried
   worker rows 8/8/2 at sched seqs 1/2/3). The emitter now fires only when
   `requested_k > 0` (real scheduled steps always ship >= 1 on a spec boot),
   making worker:sched 1:1.
2. **The C1 counter's zero delta was the async scheduler.**
   `SchedulerConfig.async_scheduling=None` AUTO-resolves to True for MTP
   boots ("mtp" is an `EagleModelTypes` member, the executor supports async,
   no incompatibility fired), so the gate boots silently ran the
   `AsyncScheduler` while P3.S7's argv-only grep reported "async flags:
   none", the same argv-only hole the round-1 review landed for the
   baseline-boot check. Under the async scheduler the engine core never
   calls `update_draft_token_ids`, so the round-4 chain's only C1 staging
   site was dead and the counter stayed at a zero delta over 23 steps /
   249 evidence samples (one-step lag cannot explain that; the transport
   was simply never fed). Fixes: the harness boots pass
   `--no-async-scheduling` (env EXTRA append) so the documented sync path
   really runs; S7 asserts ENGINE TRUTH (the patched sync channel logs once
   on its first `update_draft_token_ids` call, impossible under the async
   scheduler) instead of argv; and the C1 list ALSO rides
   `ModelRunnerOutput.sampled_per_pos`, staged by `update_from_output`
   after that step's fold, so the counter stays live on BOTH scheduler
   modes (the async scheduler is what production runs).
3. **Position-0 sample accounting moved to the prefill rejoin** (count
   `num_reqs` once per real call after the eager/captured-graph prefill
   branches rejoin, graph padding excluded, completed work, not intent).
4. **S6's probe was an ~8-token prompt**, far below one 1,600-token
   prefix-cache block, so the identical second request could never record a
   hit (the live run: hits delta 0.0). The probe now exceeds one block and
   verifies its tokenized length from the server's own `usage`.
5. **S3's mixed-step shape never fired** (one 32-token request fired 1s
   into a 9000-token prefill: the 8192-token chunk consumed the whole token
   budget and the short request finished before the long row's first
   verify). The shape now fires several longer decodes staggered past the
   long request's chunks; the check stays FAIL-strict.
6. **The run sequence is now executable**: when P4 runs and the currently
   running container is not a valid off/MTP=3 boot, the orchestrator BOOTS
   the off/MTP=3 baseline itself (a guarded scratch copy of
   `baseline_config_dir`, sync scheduling like every gate boot) and captures
   the baseline there, off/MTP=3 boot -> capture-baseline -> gate boots ->
   restore, instead of failing P4.D0.

The `selftest` is 52/52 PASS (round 4 was 45/45); the LIVE gate run is
still pending the iter7b image rebuild (this round changes
`patch_width_runner.py`, so the previously built iter7b image does not
carry these fixes).

Round 6 (2026-10-14, after the round-5 re-review) fixes the three findings
that review landed against the round-5 HARNESS changes (the two engine
root-cause fixes were confirmed correct):

1. **P3.S7 self-tripped on the harness's own flag.** Round 5 appended
   `--no-async-scheduling` to every harness boot's env EXTRA, and S7's argv
   grep matched ANY token containing "async", so on every harness boot the
   check counted the harness's OWN negated flag, `bool(sync_lines) and not
   async_flags` went False, and P3 (hence OVERALL) false-FAILed with the
   self-contradictory detail naming the exclusion flag as an async flag.
   The grep now matches the ENABLE form positively (`argv_async_flags`:
   `--async-scheduling` / `--async-scheduling=...`, underscores and case
   tolerated) and excludes the negated forms (`--no-async-scheduling` and
   friends); any OTHER async-bearing token still counts (fail-noisy).
   Selftest: the negated flag + a live sync channel -> PASS;
   `--async-scheduling` + a live sync channel -> FAIL.
2. **The auto-baseline booted the WRONG config dir.** `_boot` hardcoded
   `base_config_dir` (v18a-g2, whose committed env sets
   `ADAPTIVE_WIDTH_MODE='armed'`/`ADAPTIVE_WIDTH_FORCE='oscillate'`) as the
   copy source for the off/MTP=3 floor boot, the P4 baseline would have
   been silently captured from an ARMED boot. The copy source now lives on
   the BootSpec (`base_config_rel`; baseline -> `baseline_config_dir`/v16b,
   gate boots -> `base_config_dir`) and `prepare_config_dir` refuses a spec
   without one, exactly one source of truth. The off floor is also pinned
   EXPLICITLY in the copied env (`ADAPTIVE_WIDTH_MODE='off'`, the force
   knob emptied, which serve.sh's `-n` test treats as unset) so an
   inherited armed/oscillate line can never leak through whatever dir a
   spec copies, the dead `ADAPTIVE_WIDTH_CONFIG_FILE` env line is skipped
   on unpatched (v16b) copies, and the freshly booted floor is
   RE-VALIDATED before anything is captured (`validate_booted_baseline`:
   off mode via Config.Env, MTP=3 from env AND argv, and the
   `--no-async-scheduling` exclusion actually present in the booted argv) -
   a poisoned floor aborts the run loudly, never silently compares.
3. **Token-budget trim exemption for B4/A8 row accounting.** A request
   trimmed to 0 tokens by the token budget is absent from
   `num_scheduled_tokens` (scheduled_running_reqs only appends actually
   scheduled requests) while its row can persist in the worker batch -
   rows beyond bs with no scheduled tokens (id absent from the paired
   sched line's reqs, win == 0 by absence) are engine-legitimate. B4/A8
   (and A3c's executed-sample identity, `sum(spp) == (bs + surplus) x
   k_eff`) now exempt exactly those surplus rows, and ONLY them: every
   row the scheduler DID schedule must still appear exactly once, so a
   dropped or duplicated SCHEDULED row still fails. The exemption count is
   reported in the check detail (never a silent relaxation). Selftest:
   row-trim / osc-row-trim -> PASS with the count visible; row-drop /
   osc-row-drop -> FAIL.

The `selftest` is 61/61 PASS (round 5 was 52/52). The LIVE gate run is
still pending the iter7b image rebuild.

Round 7 (2026-10-14, after the second LIVE gate run
`runs/2026-10-14-p2-stage1b-gate/gate-run2.log`) fixes what that run exposed:

1. **P1/P2/P5 A0 false-FAILED MISSING_INSTRUMENT on a healthy boot.** The
   run's scratch `prepare.log` shows the plumbing WAS inserted
   (`SPECDBG_EVIDENCE:appended` + `serve.sh:patched(AWCFG_ARGS,SPECDBG_ARGS)`)
   and P3 - which runs AFTER P1 on the same force3 boot - PASSED its own A0,
   so the emitters and the env/serve.sh plumbing were live and working. The
   root cause is a design contradiction the round-5 emitter gate created:
   the C2 emitters are REQUEST-DRIVEN (the sched emitter fires per scheduled
   step, the worker emitter per propose with `requested_k > 0` - exactly the
   gate that removed the six boot-time JIT-warmup lines), so a FRESH boot
   has zero evidence lines until the first request runs, while A0 asserts
   "lines since boot" BEFORE any workload. The round-5-era gate runs only
   passed A0 because the OLD image still emitted the six warmup lines the
   round-5 fix eliminated - A0 was passing vacuously on lines that were
   themselves the round-5 defect. (P3's A0 passed in the same run only
   because S7's tiny probe request runs BEFORE A0.) Fix: the boot manager
   sends ONE tiny instrument probe request immediately after readiness on
   every gate boot (patch_serve specs; the off-floor baseline boot carries
   neither emitters nor plumbing and needs none), before any point runner -
   "lines since boot" is now a REAL instrumentation test (a silent or
   mis-gated emitter still fails A0, correctly), `_require_instrument`
   gained S7's bounded re-grep for the log-write race, and the failure text
   says the probe ran. Selftest: the new `no-evidence` scenario (counter
   present, emitters silent) -> MISSING_INSTRUMENT.
   Belt-and-braces (as directed): the COMMITTED `recipe/config/v18a-g2`
   serve.sh now carries the `AWCFG_ARGS`/`SPECDBG_ARGS` passthrough itself,
   BYTE-IDENTICAL to the harness's insert (the selftest asserts one
   occurrence of each insert text, that a stripped clean copy is re-inserted
   back to the committed bytes, and that the second run is a no-op), so a
   v18a-g2 boot is instrumented even if the insert were skipped. The
   committed env deliberately does NOT set `SPECDBG_EVIDENCE` (the knob is
   per-boot, set by the harness's env rewrite; default-on debug emitters in
   a recipe config would surprise every non-gate boot).
2. **The point-to-boot assignment is now explicit data, not phase blocks.**
   `POINT_PHASES` + `gate_boot_plan()` pin P1/P3/P4 -> the force3 boot
   (P4's fixed-K premise: byte-identical greedy token IDs vs the off
   baseline only holds where the proposal width is forced; an armed+oscillate
   boot is exactly the degree of freedom D1-D4 must not have) and P2/P5 ->
   the osc boot; `gate_boot_plan` refuses a plan that double-books a point,
   misses one, or pairs a point with a boot whose mode contradicts its
   premise, every point runner ASSERTS the boot it was handed
   (`_assert_boot_phase`; the selftest runs every case with the plan-assigned
   boot name so the guard is exercised everywhere), and a selftest case
   asserts the plan itself. For the record: the round-5 and round-6 code
   already executed P4 inside the force3 phase (run2's boot order and the
   code agree) - run2's P4 D1/D2/D3 failures (a one-token greedy divergence
   on the random-reject prompt, seeds 11/13 not reproducible within the
   forced boot, seed 17 at 0.61 agreement) are ENGINE-side findings for the
   next gate round, not an assignment defect.
3. **P3.S4 trigger retune + the scheduler-source analysis.** See the "S4
   trigger analysis" section below: the shape now raises concurrency (seven
   long-lived shorts, topped up between attempts) and calibrates the long
   prompt at RUNTIME onto small token-budget leftovers (measured from the
   server's own usage - never estimated), the mock's trim model is
   SHAPE-DRIVEN (chunk-boundary, not a timer), and a selftest case pins the
   untuned shape to the WARN coverage-gap report. The honest caveat from
   reading the scheduler source is in that section: the trigger may be
   structurally unreachable in this engine build; the check detail and
   `res.data["s4_shape"]` now report the calibrated lengths so the next live
   run settles it.

The `selftest` is 66/66 PASS (round 6 was 61/61). The LIVE gate run is
still pending the next coordinated boot window.

## Phase model (read before the assertions)

Quantities and their sources:

- **launch_k**, `num_speculative_tokens` at boot (gate uses 4). Sizes buffers,
  cudagraphs, and the Prometheus position labels (0..launch_k-1).
- **requested_k**, `SchedulerOutput.num_spec_tokens_to_schedule` for the step,
  after the armed branch (`patch_adaptive_width_armed.py`). Scheduler-side.
- **k_eff**, the proposal width the drafter actually samples this step.
  Worker-side. In the pre-stage-1b image there are ZERO consumers of
  requested_k on the V2 path (`vllm/v1/worker/gpu/model_runner.py` +
  `gpu/spec_decode/autoregressive/speculator.py`), so k_eff is pinned at
  launch_k: that is the defect stage-1b removes.
- **win**, per-request verify-window length =
  `len(scheduled_spec_decode_tokens[req_id])`. This is the PREVIOUS
  proposal's width (one-step lag), possibly shrunk by a budget **trim**
  (`sched-830-860`: legitimate generate>verify divergence). **Round 4
  (k_eff semantics): there is NO launch-K cold-start seeding on the sync
  path**, `request.spec_token_ids` is only ever set from the DraftTokenIds
  output (`update_draft_token_ids`, and the async-only `[-1]*launch_k`
  placeholder lives in AsyncScheduler, which is excluded per P3.S7), so a
  request's FIRST verify window is ALSO the previous proposal's width. A
  launch_k-width window on a non-first verify is exactly what the pre-1b
  launch-width carrier produces, a DEFECT the bounds/lag checks flag, not
  an invariant.
- **sampled_per_pos[pos]**, how many (request, position) draft samples the
  drafter actually executed at draft position `pos` this step. Worker-side
  truth, accumulated IN the speculator (first-pass sample + each eager
  `_generate_draft` call) and shipped both on the C1 Prometheus counter and
  on the worker evidence line. THE discriminator D-009 demands: scheduler
  counters cannot distinguish "generated short" from "generated wide then
  trimmed" (pos-3 sampled with a 3-wide verify window is the trim-defect
  signature). On this drafter every batch row samples at every position
  0..k_eff-1, rows without a scheduled verify window (chunked prefill)
  consume their samples too; only their scheduler-side windows stay empty.

Phase relations the gate asserts (t = engine step):

1. `k_eff(t) == requested_k(t)`, the drafter consumed THIS step's width.
2. `win(t) == requested_k(t-1)` per request in flight at both t-1 and t -
   the pending verification keeps its own length while the next proposal
   changes (the D-009 "current verification keeps its own lengths"
   invariant; on a healthy drafter k_eff == requested_k, so the previous
   proposal's width IS requested_k(t-1)). **Trim exemption (round 2):** a
   row with `win < requested_k(t-1)` is a legitimate budget trim (the
   phase model above; P3.S4 provokes them deliberately) and is EXEMPT from
   the lag check, not an offender. The hard invariant is
   `win > requested_k(t-1)` = violation: a window can never be LONGER than
   the previous proposal. The lag defect (window follows the CURRENT
   width) still fails through its grow-step rows; **round 4:** the pinned
   drafter (stage-1b consumption missing) ALSO fails here, its
   launch_k-width windows exceed the previous requested width on every
   shrink step (P5.E2 remains the explicit discriminator that names the
   reason).
3. `sampled_per_pos` support `== {0..k_eff-1}`, no sample at position k_eff
   (for forced K=3: one first-pass sample + two draft-loop steps, NO sample at
   position 3; positions are 0-indexed draft columns of `draft_tokens`).
4. Counter phase (round 4, transport round 5): the worker's
   `sampled_per_pos` list travels to the scheduler on TWO carriers, the
   post-step DraftTokenIds channel (sync mode only: the engine core pulls
   it AFTER `update_from_output` has already sealed the step's stats) and
   the `ModelRunnerOutput.sampled_per_pos` field, which
   `update_from_output` itself reads and stages at the END of the step,
   after that step's fold drained the previous step's list (the async
   scheduler never pulls the post-step channel, so carrier B is what keeps
   the counter live there, and production runs async). Either way the
   scheduler folds the staged list into the NEXT step's SpecDecodingStats.
   That next step is exactly the verify step of the proposal that produced
   the samples (the one-step lag), so the sampled counter and the
   drafts/accepted counters of the same stats object describe the same
   proposals; window-level counter checks tolerate the residual boundary
   skew (`window_skew_steps` in gate_config.json). The tolerance is NOT
   enlarged to absorb a dead transport: a zero counter delta against
   non-zero evidence sums is a FAIL (the round-4 live run's 249-sample
   zero was the async-starved staging site, fixed by carrier B).

## Debug-counter contract (what the stage-1b implementer must add)

The harness treats missing instrumentation as `MISSING_INSTRUMENT` (a FAIL),
never a silent pass; instrumentation that is present but MALFORMED (a JSON
value of the wrong type where the contract requires an array/object) is
`MISSING_MALFORMED_INSTRUMENT` and is type-checked without being iterated, so
a buggy emitter can never crash the gate run (round 3). The contract has
three parts; names are frozen, do not
rename without renaming `driver.py` (constants at the top).

### C1, the new Prometheus counter (implemented in the stage-1b patch)

```
name:      vllm:spec_decode_num_sampled_draft_tokens_per_pos
labels:    engine, model_name        (same per-engine labelnames as
                                      vllm:spec_decode_num_accepted_tokens)
           position                  (string "0".."launch_k-1", pre-created at
                                      init for range(num_spec_tokens), exactly
                                      like num_accepted_tokens_per_pos)
type:      Counter                   (exposition appends _total)
```

Semantics: +1 per **actually-executed drafter sample** per (request, draft
position). Concretely, in `gpu/spec_decode/autoregressive/speculator.py`
(as implemented by `patch_width_runner.py`, round 4):

- position 0 is counted at `propose()` entry: the first-pass sample
  executes for EVERY batch row later in the same call, eagerly in
  `_prefill` or inside the captured prefill graph, either way exactly one
  sample per row at draft position 0;
- positions 1..k_eff-1: one increment per request after each eager
  `_generate_draft` call, in BOTH loop variants
  (`_multi_step_decode`, `_generate_fused_drafts`);
- NOT counted: `dummy_run=True` / `is_profile=True` propose calls (the
  counter is staged to None there, so boot prewarm cannot pollute it and a
  stale list is never re-shipped), positions >= k_eff, verify or rejection
  work, the bonus token. FULL-graph decode replay would bake its samples
  into the captured graph and is NOT counted, the same physical limit
  class as the baked loop bound; this deployment runs eager draft decode
  (PIECEWISE target graphs, fused draft decode disabled for the QSA
  backend), so the counter is exact here.

`k_eff == 0` (zero-width proposal) is first-class: a real propose with a
zero width records a list of zeros, distinct from "no real propose this
step" (None). D-009: "Define zero-width behavior separately from None."

Transport (why this path: `/metrics` lives in the API-server process; worker
counters never reach it directly), as implemented, with the round-5
additions:

1. Worker: the speculator accumulates the per-position sample counts for
   the step (`awdbg_sampled_per_pos`, length launch_k, reset per real
   propose, None on dummy/profile). Position 0 is counted at the PREFILL
   REJOIN (round 5: after the eager/captured-graph prefill branches
   rejoin, `num_reqs` once per real call, graph padding excluded) -
   completed work, not intent.
2. Worker -> scheduler, channel A (sync post-step): the list rides
   `DraftTokenIds` (vllm/v1/outputs.py) via
   `DraftTokensHandler.set_draft_tokens`/`get_draft_tokens` (both paths),
   and the engine core's `post_step` hands it to
   `Scheduler.update_draft_token_ids`, **only when async scheduling is
   OFF**.
3. Worker -> scheduler, channel B (round 5, mode-independent): the SAME
   step's list also rides `ModelRunnerOutput.sampled_per_pos` (set by the
   V2 runner right after propose); `Scheduler.update_from_output` stages
   it at the END of the step, AFTER that step's fold drained the previous
   step's list, so the one-step lag (phase relation 4) holds on BOTH
   channels and each list folds exactly once (in sync mode both channels
   carry the identical list; no double count). Channel B exists because
   the async scheduler never pulls channel A: the round-4 chain's only
   staging site was dead there, which is why the round-4 live run showed a
   zero counter delta.
4. Scheduler -> frontend: `SpecDecodingStats`
   (vllm/v1/spec_decode/metrics.py) gained
   `num_sampled_draft_tokens_per_pos: list[int]` (initialized
   `[0]*num_spec_tokens` in `new()`), the scheduler FOLDS the staged list
   at the end of the next `update_from_output` (one step after the samples
   executed, see phase relation 4), and `SpecDecodingProm.observe()`
   increments the new counter per position next to the existing per-pos
   accepted counter.

Sizing note (D-009 metrics trap): with launch K=4 the per-pos accepted
counter grows to positions 0..3 and `SpecDecodingStats.observe_draft`'s
`assert num_accepted_tokens <= self.num_spec_tokens` is sized at launch K -
no change needed there, but the new counter's position range must be
launch_k, not the per-step k.

### C2, evidence lines (required; per-step, per-request detail)

Prometheus cannot carry per-step ordering or request identity. The stage-1b
patch therefore adds two env-gated emitters, both `logger.info` lines
containing the fixed tag `AWDBG1B` followed by one JSON object:

Gate: `VLLM_SPECDBG_EVIDENCE=1` (default 0 = one env read, zero overhead).
The copied config dir's `serve.sh` must pass it through, the harness's
config builder inserts that plumbing itself (idempotent, assert-guarded), so
the implementer does not need to touch serve.sh for the gate, but the env
knob must be read by BOTH emitters. Round 7 belt-and-braces: the COMMITTED
`recipe/config/v18a-g2/serve.sh` now carries the same passthrough,
byte-identical to the harness's insert (the selftest proves a stripped clean
copy is re-inserted back to the committed bytes), so the plumbing exists even
where the insert never runs; the committed env deliberately does NOT set the
knob (the harness's per-boot env rewrite does).

Worker line (emitted after each `speculator.propose` that belongs to a
SCHEDULED batch, round 5: the emitter gate is
`num_spec_tokens_to_schedule > 0`; the runner-side JIT kernel warmup runs
six real non-dummy steps whose fabricated `SchedulerOutput.make_empty()`
carries the default 0 and never reaches the scheduler, so those steps emit
nothing, per execute_model):

```json
AWDBG1B {"src":"worker","seq":<int>,"t":<unix_us>,"k_eff":<int>,
         "sampled_per_pos":[<int x launch_k>],
         "reqs":[{"id":"<req_id>","valid_len":<int>,"row":<int>,
                  "steps":[<int>]}]}
```

- `k_eff`: the per-call proposal width used (0 allowed; null never).
- `sampled_per_pos`: the SAME list that feeds the C1 counter this step -
  the speculator's own executed-sample counter, NOT a function of
  scheduler-visible windows (round 4: on this drafter that is
  `[num_reqs]*k_eff` followed by zeros; every batch row samples every
  position 0..k_eff-1, including rows without a scheduled verify window).
- `reqs[].valid_len`: the number of draft positions actually sampled for
  that request. **Round 4 (worker truth): `k_eff` for EVERY row**, the ar
  drafter samples the whole batch (first-pass sample + each
  `_generate_draft` call), so rows without a scheduled verify window
  (chunked prefill) consume their samples too; only their scheduler-side
  windows stay empty. (The round-2 model, valid_len 0 for windowless
  rows, derived the field from `num_draft_tokens_per_req`, i.e. from
  scheduler-visible state; the round-4 review rejected that as
  contradicting C1's "actually-executed" semantics, and the emitter now
  takes `sampled_per_pos` from the speculator instead of the window list.)
- `reqs[].row` (round 2, REQUIRED): the req_state row the request occupies
  THIS step, `int(input_batch.idx_mapping[j])`, the row index the proposal
  buffers are addressed by (`req_states.draft_tokens[input_batch.idx_mapping]
  = draft_tokens`, V2 runner :1907; the proposal-logit cache
  `draft_logits[max_num_reqs, num_speculative_steps, V]` is indexed
  `req_state_idx * stride_0 + draft_step * stride_1`, rej-utils :162-174).
- `reqs[].steps` (round 2, REQUIRED): the per-token draft step indices of
  that request's samples this step, ascending, each in `[0, k_eff)`, for
  every row exactly `range(valid_len) == range(k_eff)`. The kernels take
  the per-token step from `expanded_local_pos` and skip
  `draft_step_idx >= num_speculative_steps` as the bonus token
  (rej-utils :228-232), the emitter reports the non-bonus steps it wrote.
- `reqs` MUST list EVERY batch row (whatever its scheduler-side window
  state, under round-4 worker truth every row samples, and the row list is
  how the gate sees the full batch), P2.B4 asserts
  `sched bs == len(worker reqs)` as row
  accounting, and (round 3) P1.A8 asserts the same against the paired sched
  lines; a worker line with `reqs` omitted reports MISSING_INSTRUMENT and a
  wrong-typed `reqs`/`reqs[]`/`reqs[].steps` value reports
  MISSING_MALFORMED_INSTRUMENT (never a crash, see C2-alignment).
- **Order contract (round 2, REQUIRED):** `reqs` MUST be emitted in the
  scheduler's batch order (the `input_batch`/`idx_mapping` order for the
  worker line; the running-set/scheduled order for the sched line). A
  canonical (id-sorted) emitter violates this contract and blinds
  P2.B4b's order-compare, P2.B4c detects it and fails the point.

Scheduler line (emitted at the end of `Scheduler.update_from_output`, per
engine step; `scheduler_output` is in scope there):

```json
AWDBG1B {"src":"sched","seq":<int>,"t":<unix_us>,"launch_k":<int>,
         "requested_k":<int>,"bs":<int>,"mode":"<controller mode>",
         "reqs":[{"id":"<req_id>","win":<int>,"acc":<int>,"rej":<int>}]}
```

- `requested_k`: the `num_spec_tokens_to_schedule` shipped in THIS step's
  SchedulerOutput (post-armed-branch value).
- `win`: `len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))`.
- `acc`/`rej`: the per-request accepted/rejected counts already computed in
  `update_from_output` (0 when the request had no spec verify this step).
- `mode`: `str(controller.mode)` when installed, else "off".
- Order: same batch-order requirement as the worker line (above).

Pairing contract (round 5): both emitters keep their own monotonically
increasing `seq` per engine step, **per-source counters that START
TOGETHER and advance 1:1**, which holds because both emitters now count
exactly the SCHEDULED steps:

- the sched emitter fires once per `update_from_output` with a non-empty
  `num_scheduled_tokens` (the scheduled step);
- the worker emitter fires once per `speculator.propose` with
  `requested_k > 0`, and on a spec boot every scheduled step ships
  `num_spec_tokens_to_schedule >= 1` (`schedule()` seeds the field with
  `num_spec_tokens` before the armed branch), runs exactly one propose,
  and reaches the emitter; steps that never reach the scheduler (the
  runner-side JIT kernel warmup, dummy/profile runs) carry requested 0 or
  dummy flags and emit nothing.

So both counters take the value 1 on the boot's first SCHEDULED step and
pair 1:1 from there; `join_by_seq` equality is the pairing invariant, and
P1.A8b fails the point on any unpaired worker line (a WARN-class
unpaired-sched surplus is reported by P2.B4a). The round-4 contract -
"one worker line per non-dummy propose", was defective: the six JIT-kernel
warmup proposes are non-dummy but never scheduled, and their lines desynced
the two counters for the whole boot. Volume is gated by the env knob and
the gate workloads are a few hundred steps.

### C2-alignment, the D-009 proposal-logit cache alignment checks (round 2)

D-009 section b ends "Pair this with deterministic cache/request/position
alignment checks", the stale-q' defect class it demonstrated (q sampled vs
stale q' = (0.5,0.5) against p=(0.5,0.5) accepts everything and stays
greedy-identical) lives in the proposal-logit cache's (row, step) addressing.
Round 1 delivered neither the checks nor a deferral; round 2 adds them as
first-class harness checks over the `row`/`steps` fields above:

- **Row ownership** (P1.A8 / P2.B6, per step): every request reports its
  req_state row; rows are unique within a step, two requests must never
  share a row (a stale `idx_mapping` or a remap bug puts request A's reads
  on request B's writes).
- **Per-token step alignment** (same checks): a request's `steps` are exactly
  `range(valid_len)`, contiguous from column 0, no skipped or reused step
  column, every index `< k_eff`. A PERMUTED/shifted mapping leaves the
  aggregate support (`P1.A3`/`P2.B3`) and the sample-count identity
  (`P2.B4`) clean, only the per-request assert sees it (the selftest's
  `osc-column-mix` scenario proves exactly that gap).
- **Across width transitions** (P2.B6b): when `k_eff` changes between
  adjacent paired steps, every request sampling in the later step must fully
  re-cover `0..k_eff-1`, the grow-back step is where a stale column from an
  earlier wider proposal would be reused instead of rewritten.
- **Row remaps are legitimate** (P2.B6c, WARN-if-never coverage): requests
  MAY move rows between steps (idx_mapping is re-derived per step); moving
  is not the defect, sharing and skipping are.
- **Epistemic boundary (honest, tightened round 4):** these are WRITE-side
  assertions, and after the round-4 emitter change they split into two
  classes. The **row-ownership half is an engine observation**, `row` is
  `int(input_batch.idx_mapping[j])` read straight off the runner's state,
  and two requests sharing a row is a real addressing defect. The
  **steps == range(valid_len) half is EMITTER ARITHMETIC**: the round-4
  emitter derives valid_len and steps from k_eff by convention (every row
  samples 0..k_eff-1 on this drafter), so this half proves the emitter's
  reporting convention, not the engine's per-token addressing, it is
  structurally unable to fire on the live path (the selftest's
  osc-column-mix defect is injected INTO the emitter output, which the real
  emitter cannot produce). Value-level staleness (sampled q vs stale q'
  content) remains P4.D4's distribution gate; the kernel-level
  block-verifier gate stays on the manual list below.
- An emitter without `row`/`steps` predates this contract and reports
  `MISSING_INSTRUMENT` on the alignment checks, never a silent pass.
- **Round 3, the reqs payload itself is required and type-checked
  (never a crash):** a worker line that omits `reqs` entirely reports
  `MISSING_INSTRUMENT` on the alignment checks; a `reqs` value (or a
  `reqs[]` row, or a `reqs[].steps` value) of the wrong JSON type reports
  `MISSING_MALFORMED_INSTRUMENT` (a FAIL-class sibling of
  `MISSING_INSTRUMENT`: the emitter exists but is emitting the wrong
  shape). Malformed values are type-checked WITHOUT being iterated, so a
  buggy emitter fails its check instead of killing the whole gate run with
  a traceback (the round-2 re-review's N5: `reqs[].steps: 5` crashed
  `run_p2` at the steps parse).
- **Round 3, P1 row accounting (the N6 fix):** P1.A8 receives the paired
  sched lines and asserts `sched bs == len(worker reqs)` exactly like
  P2.B4, an emitter that drops the reqs payload (or emits `reqs: []`
  under a non-empty batch) now fails P1 loudly instead of passing
  vacuously over zero rows. (P2 keeps B4 as its row-accounting check; B6
  there does not duplicate it.)
- **Round 6, token-budget trim exemption in row accounting:** a request
  the token budget trims to 0 tokens is absent from `num_scheduled_tokens`
  (scheduled_running_reqs only appends actually scheduled requests) while
  its row can persist in the worker batch and sample like every row
  (worker truth). Rows beyond bs whose id the paired sched line's reqs do
  NOT contain are exempt from the `sched bs == worker rows` equality in
  P1.A8/P2.B4, and A3c's executed-sample identity covers them too
  (`sum(spp) == (bs + surplus) x k_eff`). The exemption is bounded: every
  row the scheduler DID schedule must still appear exactly once, so a
  dropped or duplicated SCHEDULED row still fails, and the exempt-row
  count is reported in the check detail (never a silent relaxation). The
  selftest proves both sides: row-trim / osc-row-trim pass with the count
  visible; row-drop / osc-row-drop still fail.

### Epistemic disclosure, engine observations vs emitter arithmetic (round 4)

Which gate inputs observe the engine and which only re-derive the emitter's
own arithmetic. Before C1 landed, `sampled_per_pos`/`valid_len`/`steps` on
the worker line were all derived from scheduler-visible state
(`num_draft_tokens_per_req`), which made several checks tautological
against the live emitter, the round-4 review's D-009 attack. After C1:

- **Engine observations** (a defect in the engine can move them):
  `sampled_per_pos` on the worker line AND the C1 Prometheus counter (both
  are the speculator's executed-sample counter, accumulated at the
  first-pass sample and after each eager `_generate_draft` call), `k_eff`
  (= `draft_tokens.shape[1]`, genuine worker truth), `reqs[].row`
  (= `int(input_batch.idx_mapping[j])`), the req ids and row counts,
  `requested_k`/`launch_k`/`mode`, and the sched line's `win`/`acc`/`rej`
  (scheduler state). The trim discriminator survives through `k_eff` and
  the counter, not through the window list.
- **Emitter arithmetic** (a reporting convention; structurally unable to
  fire on the live emitter): `valid_len` and `steps` are derived from
  `k_eff` by convention (`valid_len == k_eff`, `steps == range(k_eff)` for
  every row), so the `steps == range(valid_len)` half of P1.A8/P2.B6 and
  the per-token half of B6b prove the emitter's reporting convention, not
  the engine's per-token addressing. The row-ownership half of those checks
  stays an engine observation (see C2-alignment).
- **Still value-blind**: no sample-EVENT counter can see a stale column
  READ without a fresh sample; value-level staleness remains P4.D4's
  distribution gate, and the kernel-level block-verifier gate stays manual
  (below).

### C2-fields, the round-8 field completion (D-014 condition 2, the astra (b) design)

The stage-1 closure completes the emitter record so the harness can join
PROPOSALS (worker side) with VERIFY EVENTS and COMMITS (scheduler side) per
request. Image iter7f applies `patch_c2_fields.py` on top of iter7d; all
fields are ADDITIVE, an old-shape line stays parse-compatible and the new
checks report MISSING_INSTRUMENT on absent fields, MISSING_MALFORMED_
INSTRUMENT on wrongly-typed values, never a crash.

Worker line gains (per scheduled propose, the PROPOSAL side):

- `branch`: the executed draft-sampling path, `"gumbel"` (the
  proposal-logit-cache probabilistic path, the branch `sample_draft` takes
  whenever `draft_logits is not None`) or `"greedy"` (the argmax path);
  recorded at propose entry in autoregressive/speculator.py from the SAME
  condition the sampler tests.
- `nt` / `pf`: the step's execution-shape fingerprint (scheduled token
  count; chunked-prefill presence).
- `reqs[].temp` / `seed` / `seed_set`: the drafter's EFFECTIVE per-row
  sampling temperature / RNG seed / explicit-seed flag, read from the
  sampler's CPU mirrors (`SamplingStates.temperature.np` / `.seeds.np` /
  `.seeds_set`) at the req_state row the proposal buffers are addressed by
 , NO GPU sync; a seed the user did not set is the engine's random filler
  flagged `seed_set=false`.
- `reqs[].props`: the drafter's actual proposal token ids per row, in
  draft-step order, the k_eff stored columns. ONE evidence-gated
  GPU→host transfer per scheduled step (the DraftTokensHandler placeholder
  path never carries real ids on gate boots).

Sched line gains (per scheduled step, the VERIFY/COMMIT side, one step
after the proposal it verifies):

- `reqs[].rej_idx`: the first rejected position of the verify (= the
  accepted count; the window length when every position accepted, the
  bonus position). null without a verify window.
- `reqs[].bonus`: the target's own sampled token at rej_idx (the
  replacement on a rejection, the bonus token on a full acceptance).
- `reqs[].emit`: the tokens actually committed this step (exactly what
  entered the request's output list, stop-trims reflected; `[]` on prefill
  chunk steps).
- `reqs[].stopped`: whether THIS commit stopped the request, `bool(stopped)`
  as of the stop-check chain (the `_update_request_with_output` stop-trim or
  the pooling/encoder stop arms; the later grammar-error stop is not part of
  this field). Recorded on every commit row; the stop-trim repair consumes it
  as the termination proof for a shortened emit tail. ADDITIVE (absent on
  old-shape lines).
- `reqs[].phash`: committed-prefix hash, sha256 over the request's WHOLE
  committed token prefix (prompt + output, int64-le encoding) AFTER the
  append, maintained incrementally per request
  (`awdbg1b_evidence.commit_hash_update`; re-based from scratch if the
  committed length ever regresses; pruned to the live request set).
- `reqs[].ol` / `nct`: the semantic cache-state fingerprint pair after the
  step (committed output length; num_computed_tokens post-rollback).
- `reqs[].temp` / `seed` / `seed_set`: the sampling params AS SCHEDULED
  (the scheduler-side mirror the join cross-checks against the worker's).

Harness checks (round 8; run in P1 as `P1.C2.*` and in P2 as `P2.C2.*`):

| check | assertion |
|---|---|
| `*.C2.fields_worker` | every worker line carries branch/nt/pf + per-row temp/seed/seed_set/props, typed |
| `*.C2.fields_sched` | every sched row carries rej_idx/bonus (null-able) + emit/phash/ol/nct + the sampling mirror, typed |
| `*.C2.proposal_join` | every sched step with a verify window pairs with the worker step one seq EARLIER (the one-step lag); the series' first sched line is a boundary skip; any later missing producer FAILs |
| `*.C2.argmax_greedy` | on greedy rows (temp == 0): `emit[:rej_idx] == props[:rej_idx]` (the greedy verify accepts exactly the drafter's argmax prefix), `len(emit) == rej_idx+1`, `emit[rej_idx] == bonus`, and a true rejection resamples a DIFFERENT token (`bonus != props[rej_idx]`), the target-logit/argmax evidence; probabilistic rows are exempt |
| `*.C2.prefix_hash` | per request the chain grows by exactly `len(emit)` and the hash moves on growth (a regression re-bases) |

Epistemic split (same discipline as C2-alignment): `props`/`branch`/
`temp`/`seed` are ENGINE OBSERVATIONS (the drafter's own buffers and
executed branch); `rej_idx`/`bonus`/`emit` are derived from the scheduler's
verify accounting (the same numbers the acc/rej counters fold); `phash` is
emitter arithmetic over the engine's committed list, its VALUE is engine
truth, its incremental maintenance is the emitter's contract, and the
cross-boot comparison (P4.D5) is only meaningful between same-shape boots.

### AWFP, the runtime behavioral fingerprints (D-014 condition 1)

Image iter7f applies `patch_fingerprints.py`: marlin_moe.py gains the
`qwen38-awfp` block (the import-time CODE check) and gpu/model_runner.py
gains the load_model install (CONFIG + the behavioral PROBE) and the
scheduled-step POLICY emit. Every line carries the tag `AWFP` + one JSON
object with a `kind` field:

| kind | content | when |
|---|---|---|
| `CODE` | the marlin module's own load-time state: the two `moe_wna16_marlin_gemm` call sites hard-set the atomicAdd tail flag false (THE atomic_add=0 proof for the Marlin MoE GEMMs, the drafter's GEMM path; the dense-marlin wrapper's env-conditional gate is recorded separately), the stable-sort shim marker + call site present, the shim env-gate state | marlin_moe import, once per process |
| `CONFIG` | versions, the kernel-selection env gates, the spec-decode config, cudagraph mode + configured capture sizes, the marlin code-state | model-load end |
| `PROBE` | the BEHAVIORAL verdict: a crafted non-canonical within-group order must come back canonical (i) through the shim helper directly and (ii) through the REAL composition (`moe_align_block_size` + the shim, the per-forward Marlin MoE sequence); the shim's own debug checker is the asserter. A shim that is absent, unwired or env-gated off fails LOUDLY | once per worker process, at load end |
| `POLICY` | the cache/kernel-selection policy hash: triton autotune selections (the fla/GDN kernels), the `VLLM_TRITON_FORCE_FIRST_CONFIG` pinning state, the inductor compile-cache entry count (informational, NOT hashed, the cache mount persists and grows), cudagraph configured+captured sizes → one `policy_sha`; the full components ride the line so a drift can be diagnosed from the log | scheduled steps 1 and 32 (post-warmup; late captures covered) |
| `DRIFT` | this boot's policy_sha differs from the previous boot's persisted record (`/root/.cache/q38-awfp/`, best-effort, needs CACHE_MOUNT), the autotune boot-lottery (R-029) fired, reported loudly | on a policy change |
| `FAIL` | an internal fingerprint failure (stage named); the fingerprint never breaks the boot | any stage |

Harness checks (round 8): `P1/P2/P4.FP.code_state` (CODE records present
and ok), `*.FP.probe` (PROBE present and passed, shim env on),
`*.FP.failures` (no AWFP-FAIL line), and `P4.FP.cross_boot_policy` (this
boot's policy_sha vs the baseline capture's `awfp_policy_sha`, same-shape
pairs only; a shape-mismatched legacy pair skips with the reason stated; a
mismatch FAILs with the pin named). A boot of a pre-iter7f image reports
MISSING_INSTRUMENT on the FP checks, the fingerprint layer's absence is
itself loud, never a silent pass. Opt-out: `VLLM_AWFP=0` (an opted-out
boot is MISSING_INSTRUMENT by design).

### C3, boot env the harness sets (implementer reads, harness writes)

| env | meaning |
|---|---|
| `VLLM_ADAPTIVE_WIDTH_MODE` | off/shadow/armed/force (existing) |
| `VLLM_ADAPTIVE_WIDTH_FORCE` | `3` (P1/P3/P4 boots), `oscillate` (P2/P5 boots) |
| `VLLM_ADAPTIVE_WIDTH_CONFIG` | set to `/aw-gate-config.json` (mounted by the harness's serve.sh insert) |
| `VLLM_SPECDBG_EVIDENCE` | `1` enables both C2 emitters |

The harness generates `aw-gate-config.json` with
`candidate_steps = [3, 4]` in the TOP-LEVEL config AND in EVERY tier
(`tiers: {"1": {"candidate_steps": [3, 4]}}`).

**D-009 config trap 1 (the harness handles it; do not undo it):** the
defaults are `candidate_steps [3,7]`. Since the round-1 controller fix
(commit 157aadf) `install(launch_k=...)` COORDINATES the ladder with the
launch K instead of refusing: the configured ladder, top-level AND every
tier, is clamped to `<= launch_k` with a logged warning, a scope with
nothing left at or below it derives the single-candidate ladder
`[launch_k]`, and `launch_k` is REQUIRED for the driving modes (armed/force;
the armed wiring passes `launch_k=self.num_spec_tokens`). The harness still
writes `candidate_steps [3,4]` in BOTH the top-level config and EVERY tier
so the gate boots need no coercion, and asserts the install log line shows
`candidates=[3, 4]` and `launch_k=4` on every gate boot; a missing install
line is a FAIL ("controller fell back to static K"), not a silent pass.

**D-009 config trap 2:** numeric `FORCE` is honored only in `mode=force`,
parsed strictly (stripped, then ASCII digits only, ` 3 `/`+3` leniency is
refused), and must be `>= 1`: zero/negative is a config error on this stack
(armed maps a zero slot to config_k; the launch pins K >= 1) and is refused
at install. Armed mode only special-cases `"oscillate"`. Hence the boot
matrix below: force boots use `MODE=force FORCE=3`; the oscillation boots
use `MODE=armed FORCE=oscillate` (the square wave then flows through the
armed `k_for_step` branch, which is the point of P5).

## The five points as concrete assertions

Check IDs are what `driver.py` emits and what the JSON result file carries.
Statuses: PASS / FAIL / WARN / SKIP / MISSING_INSTRUMENT
(+ `MISSING_MALFORMED_INSTRUMENT` on individual checks, a FAIL-class sibling
for instrumentation that is present but of the wrong JSON type; it
aggregates as `MISSING_INSTRUMENT` at point level).

### P1, direct-work evidence (boot `force3`: launch_k=4, MODE=force FORCE=3)

| check | assertion |
|---|---|
| `P1.A0.instrument` | new counter present in /metrics AND `AWDBG1B` worker+sched lines present since boot; else MISSING_INSTRUMENT. Round 7: the boot manager's instrument probe (one tiny request right after readiness) guarantees at least one scheduled step exists before A0 asserts - the C2 emitters are request-driven, so on the probe-less boots of the 2026-10-14 round-7 run this check false-FAILED P1/P2/P5 on healthy boots; the failure text now says the probe ran, so a MISSING_INSTRUMENT is a real emitter fault |
| `install.*` | the install log line exists with `mode=force force=3 candidates=[3,4] launch_k=4` (failed install = silent static-K fallback = FAIL) |
| `P1.A2.requested_k` | every sched line in the scrape window has `requested_k == 3` |
| `P1.A2b.launch_k` | every sched line has `launch_k == 4` |
| `P1.A3.sampled_support` | sampled-per-pos deltas: support exactly {0,1,2} |
| `P1.A3b.no_sample_at_K` | delta(samples at position 3) == 0, THE trim discriminator: "generated 4 and trimmed 1" would show pos-3 samples behind a 3-wide scheduler window |
| `P1.A3c.k_eff_matches_request` | round 4 (uniform executed sampling): observed k_eff (max support + 1) == 3, AND `sum(sampled_per_pos) == bs x k_eff` on every worker line with bs from the PAIRED sched line (independent source), the ar drafter samples EVERY batch row at every position 0..k_eff-1, so a sampled_per_pos derived from scheduler-visible windows (the pre-C1 fabrication) breaks this identity on every step that mixes windowless rows in; AND the counter-vs-evidence reconcile: delta counter sum(samples) vs the evidence lines' sum(spp), within the boundary-skew tolerance (`window_skew_steps` x force_k x max bs, the C1 fold lands one step after the samples). The old `sum(samples) == force_k x delta(num_drafts)` form was retired: rows sample on their prefill/first-decode steps BEFORE their first verify window exists, so samples and verify events diverge at window edges by design |
| `P1.A4a.window_counter_reconcile` | delta(`vllm:spec_decode_num_draft_tokens_total`) == Σ win rows in the evidence window, within a boundary-skew tolerance (`window_skew_steps` x launch_k x max bs) for scrape-vs-evidence window edges |
| `P1.A4b.window_bounds` | every win <= max(requested_k(t-1), launch_k), a window is never longer than the previous proposal; smaller wins (trims) allowed. Round 4: the old "win == launch_k only on a request's first verify (cold-start invariant)" clause is GONE, there is no launch-K cold-start seeding on the sync path, and a launch_k-width window on a non-first verify is exactly the pre-1b launch-width carrier, which this bound already flags on every shrink step |
| `P1.A5.no_launch_only_samples` | sampled deltas at positions 3..launch_k-1 == 0 |
| `P1.A6.valid_lengths` | every worker req has `valid_len` in (0, 3), round 4: rows WITHOUT draft positions (valid_len == 0) are exempt exactly like A3c models them; the offender is a row whose valid_len is neither 0 nor the forced width (on this drafter the emitter reports k_eff for every row, so the offender class is a width mismatch) |
| `P1.A6b.k_eff` | every worker line has `k_eff == 3` |
| `P1.A7.accepted_pos_reconcile` | accepted-per-pos deltas at positions >= 3 == 0 (the -1 tail is never accepted) |
| `P1.A8.alignment_rows` | D-009 proposal-logit cache alignment, per step: reqs[].row present and unique (row ownership), reqs[].steps == range(valid_len) with every step < k_eff (per-token draft_step_idx bounds); the check still admits rows without draft positions (valid_len 0 / steps []) for drafter generality, but the round-4 emitter reports worker truth, every row samples, so valid_len == k_eff everywhere (see C2-alignment for the epistemic split). Round 3: the reqs payload is REQUIRED (omitted -> MISSING_INSTRUMENT; a wrong-typed reqs/row/steps value -> MISSING_MALFORMED_INSTRUMENT, never a crash) and the worker row count is cross-checked against sched bs (row accounting, like P2.B4) so a dropped reqs payload cannot pass vacuously |
| `P1.FP.code_state` / `P1.FP.probe` / `P1.FP.failures` | round 8 (D-014 cond. 1): the boot's runtime fingerprints exist and prove the moe-align shim + the atomic_add call-site state behaviorally (see AWFP); a pre-iter7f image is MISSING_INSTRUMENT here |
| `P1.C2.fields_worker` / `P1.C2.fields_sched` / `P1.C2.proposal_join` / `P1.C2.argmax_greedy` / `P1.C2.prefix_hash` | round 8 (D-014 cond. 2): the C2 field completion, the proposal/verify/commit join, the greedy target-logit/argmax evidence and the committed-prefix hash chain (see C2-fields) |

### P2, in-process transitions (boot `osc`: launch_k=4, MODE=armed FORCE=oscillate, candidates [3,4])

One process, no reboot: the oscillate square wave flips requested_k between
4 and 3 at every observe() boundary (= one scheduler step). Workload:
staggered overlapping waves (bs up to 4, different max_tokens per slot) so
the running set reorders between flips.

| check | assertion |
|---|---|
| `P2.B1.square_wave` | requested_k alternates; >= 3 full cycles (flips/2) in the window |
| `P2.B2.one_step_lag` | for every request verified at both t-1 and t: `win(t) == requested_k(t-1)`, the pending verification keeps its own length while the next proposal changes. Budget trims (win < requested_k(t-1)) are EXEMPT as legitimate (phase model / P3.S4); the violated invariant is win > requested_k(t-1) |
| `P2.B3.samples_follow_current` | every worker line: support == {0..k_eff-1} and k_eff ∈ candidates. Round 4: a WINDOWLESS step (the paired sched line scheduled no draft positions on any row, the boot's first scheduled batch is one) is EXEMPT from the support assertion, there is no window coupling to assert there, and the drafter's samples on such a step are worker truth; k_eff must still be a legal ladder candidate everywhere |
| `P2.B4a.pairing` | sched/worker seq pairs complete (unpaired lines = WARN) |
| `P2.B4.reorder_identity` | per step: Σ sampled_per_pos == Σ valid_len over the worker's rows (round 4 worker truth: EVERY row samples, so both sides equal bs x k_eff; the valid_len==0 exemption for rows without draft positions stays in the check for drafter generality but cannot fire on this drafter); every sampling row has valid_len == k_eff (uniform); sched bs == worker row count (the worker line lists EVERY batch row). Aggregate identity survives bs>1 + reordering |
| `P2.B4b.reorder_observed` | the running set changes >= once between steps, order-compare (adjacent ordered id tuples differ; needs the pinned batch-order emitter) OR set-compare (adjacent id sets differ; emitter-independent). WARN only if neither fired |
| `P2.B4c.emitter_order_contract` | the sched reqs order is NOT id-sorted on every multi-row step while bs>1, a canonicalizing emitter blinds order-compare and violates the pinned C2 order contract (MISSING_INSTRUMENT when detected) |
| `P2.B5.no_stale_suffix` | zero sample events at positions >= 3 while k_eff == 3, a shrink must invalidate the suffix; stale columns leaking into later proposals is the D-009 stride/lifetime hazard. Round 4 epistemic note: with C1 landed the sampled_per_pos on the worker line IS the speculator's executed-sample counter, so this is now an ENGINE observation of sample EVENTS, it fires if the drafter ever samples beyond its declared width; it cannot see VALUE-level staleness (a stale column read without a fresh sample), which stays P4.D4's distribution gate |
| `P2.B6.alignment_rows` | D-009 alignment during oscillation, per step: row ownership unique, reqs[].steps == range(valid_len), steps < k_eff (see C2-alignment); a wrong-typed reqs/reqs[]/reqs[].steps value is MISSING_MALFORMED_INSTRUMENT, type-checked without iteration, never a run-killing crash (round 3) |
| `P2.B6b.transition_alignment` | across every k_eff change between adjacent paired steps: full re-cover (steps == range(valid_len) in the later step), no stale step column reused across the transition. SKIP when the window has no transition (a pinned drafter is P5.E2's defect and, round 4, also fails B2's lag bound; B1 covers oscillation presence) |
| `P2.B6c.row_remap_observed` | at least one request changed req_state rows across steps (idx_mapping re-derived per step; moving is legitimate, sharing is not). WARN if the workload never remapped |
| `P2.FP.code_state` / `P2.FP.probe` / `P2.FP.failures` | round 8 (D-014 cond. 1): the runtime fingerprints on the armed-oscillation boot (see AWFP) |
| `P2.C2.fields_worker` / `P2.C2.fields_sched` / `P2.C2.proposal_join` / `P2.C2.argmax_greedy` / `P2.C2.prefix_hash` | round 8 (D-014 cond. 2): the C2 field completion during oscillation, the join spans width transitions; the argmax evidence covers the greedy rows of every width (see C2-fields) |

### P3, boundary coverage (boot `force3`)

| check | scenario + assertion |
|---|---|
| `P3.S7.async_excluded` | round 5 (engine truth, not argv): after one tiny probe request (which guarantees the sync channel has run), the patched scheduler's one-time `update_draft_token_ids` log line (`sync draft-token channel live`) must be present since boot, it cannot appear under the async scheduler, AND the container argv carries no async-scheduling ENABLE flag (secondary signal only). The round-4 argv-only grep reported "async flags: none" while `SchedulerConfig.async_scheduling=None` had silently auto-resolved to True for the MTP boot and the AsyncScheduler was serving. The harness boots now pass `--no-async-scheduling` (env EXTRA append) so the documented exclusion is real. Round 6: the argv grep matches the ENABLE form (`--async-scheduling`) and excludes the negated forms, the harness's OWN `--no-async-scheduling` boot flag must not self-trip the check (the round-5 review: the two round-5 changes were not reconciled, so every harness boot false-FAILed P3) |
| `P3.S1.first_proposal_coldstart` | round 4 (k_eff semantics): a request first seen in the window verifies with win == its first proposal's OWN width == forced K=3 (one-step lag; there is NO launch-K cold-start seeding on the sync path, `request.spec_token_ids` is only ever set from the DraftTokenIds output). Asserting win == launch_k here passed on the pre-stage-1b defective engine (launch-width carrier) and failed the correct one, inverted discrimination, fixed |
| `P3.S1.no_launch_only_samples` | sampled deltas at positions >= 3 stay 0 around the first proposals |
| `P3.S2.chunked_first_window` | a >8192-token prompt (max_num_batched_tokens) chunks; its first verify window equals its first kept proposal's width (== forced K, same round-4 model as S1, the chunked request's proposals are discarded while it is a prefill chunk, so its first verify lands one lag after its last chunk); request completes |
| `P3.S3.mixed_steps_seen` | >= 1 step mixes spec-decode rows (win>0) with non-spec rows (win=0) and the aggregate identities hold. Round-5 shape: three longer decodes (192 tokens) staggered past the long request's chunks, the round-4 shape (one 32-token request 1s into a 9000-token prefill) produced 0 mixed steps live because the 8192-token chunk consumed the whole token budget. FAIL-strict; a still-silent trigger is a reported coverage gap, not a relaxed threshold |
| `P3.S4.shortened_windows` | round 7 shape: seven long-lived concurrent decodes plus a long prompt CALIBRATED AT RUNTIME (measured from the server's own usage) onto small token-budget leftovers, up to three attempts at neighbouring leftover targets, stopping at the first trimmed window; produces win < requested_k(t-1) rows. WARN if the trigger never fires (coverage gap, not a correctness failure) - the WARN detail now names the calibrated shape, and `res.data["s4_shape"]` carries the measured probe/candidate token counts. See "S4 trigger analysis" for why the trigger may remain structurally unreachable in this engine build |
| `P3.S4b.trim_within_k_eff` | every trimmed row satisfies win <= k_eff (the scheduler never schedules more than the drafter produced) |
| `P3.S5.rejection_observed` | a low-acceptance workload (gibberish prompt, temperature 1) yields acceptance rate < 0.9 x (k_eff-1) with the per-position profile |
| `P3.S6.prefix_reuse` | round 5: the identical prompt now exceeds one 1,600-token prefix-cache block (target 3,000 long_ctx tokens ≈ ~1.9k tokenized); the tokenized length is verified from the server's own `usage.prompt_tokens` (a probe below the block threshold is a probe-shape FAIL, never a vacuous pass) and the identical second request must record a positive hit (prefix-cache hits counter delta > 0 OR `usage.prompt_tokens_details.cached_tokens` > 0). The round-4 ~8-token probe could never hit (live: hits delta 0.0) |
| `P3.S6.successor_coldstart` | round 4 (k_eff semantics): the slot successor's first verify window equals its own first proposal's width (== forced K), NOT the predecessor's width and NOT launch_k |

### P4, token-ID comparison vs the MTP=3 baseline + stochastic gate (boot `force3`)

Baseline = the MTP=3 / mode=off floor boot (the currently running config).
Captured with `capture-baseline` (same prompts, same seeds, streaming
`return_token_ids: true`, token IDs, not text bytes).

| check | assertion |
|---|---|
| `P4.D0.baseline` | baseline capture exists (greedy + seeded + accept_profile) |
| `P4.D1.greedy_token_identity` | temperature 0, all 5 prompts: completion token IDs IDENTICAL to the baseline (primary exactness assert; no tolerance). A divergence means the width plumbing changed the proposal path |
| `P4.D2.seeded_reproducible` | same seed twice within the forced-3 boot -> identical token IDs (per-request RNG determinism survives the width plumbing) |
| `P4.D3.cross_config_agreement` | same seeds, forced-3 vs baseline, temperature 0.7: per-seed token agreement >= 0.98 (exact equality is expected; small float-level divergence through different capture shapes is tolerated) |
| `P4.D4.distribution_gate` | per-position acceptance-rate profile of this boot vs the baseline profile: max delta <= 0.15, the stochastic gate that catches the D-009 defect class greedy identity misses (q sampled vs stale q' = (0.5,0.5) with p=(0.5,0.5) accepts everything and stays greedy-identical) |
| `P4.FP.code_state` / `P4.FP.probe` / `P4.FP.failures` | round 8 (D-014 cond. 1): the boot's runtime fingerprints exist and prove the shim + atomic_add fixes (see AWFP) |
| `P4.FP.cross_boot_policy` | round 8: this boot's cache/kernel-selection policy_sha vs the baseline capture's, identical on same-shape matched controls; a shape-mismatched legacy pair skips with the reason stated; a mismatch FAILs (the autotune boot-lottery, R-029; pin: VLLM_TRITON_FORCE_FIRST_CONFIG) |
| `P4.D5.prefix_hash_identity` | round 8 (D-014 cond. 2): the C2 committed-prefix chains (prompt + committed output, sha256) of the greedy prompts are byte-identical to the baseline capture's at every common committed-output length, the ENGINE-SIDE identity signal behind quantity A; same-shape pairs only (a shape-mismatched legacy baseline skips with the reason stated; the revised contract's off@K4 vs force4 pair compares fully) |

### P5, armed-drive verification (boot `osc`)

| check | assertion |
|---|---|
| `P5.E1.armed_live` | install line `mode=armed` present AND the armed-driving log line (`adaptive width armed: driving per-step K`) fired |
| `P5.E2.drive_reaches_drafter` | requested_k oscillates AND worker k_eff follows it (joined by seq). Failure discrimination is explicit: scheduler oscillation with a pinned drafter width (k_eff flips == 0) fails with "stage-1b consumption MISSING", that is exactly the pre-stage-1b state |
| `P5.E3.lag_consistency` | the P2.B2 lag relation holds during oscillation |
| `P5.E4.ema_pure` | optional separate boot (mode=armed, FORCE unset) with an acceptance-cliff workload; SOFT (WARN) because EMA timing (warmup 10 + interval 5) depends on traffic. D-009 accepts armed oscillation as the primary drive evidence |

## Boot matrix

| boot | config (scratch copy of) | MTP | adaptive | used by |
|---|---|---|---|---|
| baseline | `recipe/config/v16b` | 3 | off | P4 baseline; restore target |
| force3 | `recipe/config/v18a-g2` | 4 | force, FORCE=3 | P1, P3, P4 |
| osc | `recipe/config/v18a-g2` | 4 | armed, FORCE=oscillate | P2, P5 |

**Run sequence (round 5, executable end to end; round 7 adds the per-boot
instrument probe):**

```
off/MTP=3 baseline boot  ->  capture-baseline  ->  force3 boot (P1,P3,P4)
                         ->  osc boot (P2,P5)  ->  restore the operator's
                         baseline config
```

Round 7: every gate boot (the patch_serve specs) runs ONE tiny instrument
probe request immediately after readiness, before any point runner - the
request-driven C2 emitters emit nothing on a fresh boot, and A0's "lines
since boot" needs at least one scheduled step to be a meaningful test. The
point-to-boot assignment is explicit data (`POINT_PHASES` + `gate_boot_plan`:
P1/P3/P4 -> force3, P2/P5 -> osc), asserted when the plan is built and again
inside every point runner.

If the currently running container is ALREADY a valid off/MTP=3 boot, the
orchestrator captures the baseline from it in place. If it is NOT (the
2026-10-14 run found a shadow boot in place and P4 failed D0 with no
baseline), the orchestrator now BOOTS the off/MTP=3 baseline itself, a
guarded scratch copy of `baseline_config_dir` (round 6: the copy source is
declared per boot on the BootSpec, `base_config_rel`; the round-5 build
hardcoded `base_config_dir` here, which would have booted the ARMED
v18a-g2 env as the floor) with the same env rewrites every gate boot gets,
`--no-async-scheduling` included so the P4 comparison stays same-mode -
RE-VALIDATES the boot before capturing anything (`validate_booted_baseline`:
off mode via Config.Env, MTP=3 from env AND argv, and the sync exclusion
actually present in the booted argv; a poisoned floor aborts the run
loudly, never silently compares), captures there, and continues. The final
restore still reboots the operator's UNMODIFIED baseline copy. Every
harness-driven boot (scratch copies only; the repository config dirs are
never edited):

- copies the boot's OWN `base_config_rel` config dir into
  `runs/gate-stage1b-scratch/config-<name>/` (round 6: the source lives on
  the BootSpec and `prepare_config_dir` refuses a spec without one),
- rewrites `env` with anchored rewrites (every rewrite asserts its match
  count), including the round-5 append of `--no-async-scheduling` to the
  `EXTRA` passthrough (the sync-scheduler exclusion the D-009 contract
  demands; without it `SchedulerConfig.async_scheduling=None` auto-resolves
  to True for MTP boots and the gate silently exercises the wrong path).
  Round 6: an off-mode spec pins `ADAPTIVE_WIDTH_MODE='off'` and empties
  the force knob explicitly, so an inherited armed/oscillate line can never
  leak through whatever dir the spec copies,
- writes `aw-gate-config.json` (round 6: the matching
  `ADAPTIVE_WIDTH_CONFIG_FILE` env line is written only when the serve.sh
  insert will actually mount it, an unpatched v16b copy would otherwise
  carry dead config),
- idempotently inserts the `AWCFG_ARGS`/`SPECDBG_ARGS` plumbing into the
  COPIED serve.sh (bash -n checked before writing), EXCEPT the baseline
  boot, whose v16b serve.sh predates the AW anchor block the insert anchors
  on and which needs neither insert (explicitly logged as
  `serve.sh:skipped(...)` in the copy's `prepare.log`, never silently
  skipped). The selftest exercises this builder against the real v18a-g2
  AND v16b files.

Image pinning: the copied dir inherits v18a-g2's `image` file
(`qwen38-flash-dgx:iter7b-20261013`, the stage-1b image the implementer will
build). Until that image exists, `run` fails cleanly at `docker image
inspect`; set `boot.image_override` in gate_config.json to point at the
actual built tag (e.g. an iter7a build with the stage-1b patch applied for
a pre-merge gate rehearsal).

## S4 trigger analysis (round 7; read before touching the shape again)

What the check needs: one sched line whose row has `0 < win <
requested_k(t-1)`. Where that can come from, per the engine source
(`v1/core/sched/scheduler.py`, pristine copy archived at
`runs/2026-10-14-p2-stage1b-verify/pristine/scheduler.py`, iter7a line
numbers):

- The running loop schedules RUNNING requests FIRST (waiting/prefill after):
  for a decode row, `num_new_tokens = num_tokens_with_spec +
  num_output_placeholders - num_computed_tokens` (== the pending window
  width, ~3 on a forced-K boot), then `min(num_new_tokens, token_budget,
  input_budget - draft_slots)`; the window is TRIMMED by
  `spec_token_ids[:num_scheduled_spec_tokens]` when that min cut into it, and
  a row whose `num_new_tokens` lands at 0 is skipped entirely (the round-6
  zero-trim/absent shape). A partial trim therefore needs
  `input_budget - draft_slots` to be nearly exhausted by the rows AHEAD of
  the decode row in `self.running` order.
- `input_budget` = `max_num_batched_tokens` (8192 in this recipe) and the
  only workload-shaped consumer that big is a prefill chunk. But promotion
  into `self.running` happens in the waiting loop AFTER the running loop,
  which on a full-budget chunk step has zero slack
  (`if input_budget <= draft_slots: break`) - so a decode row with a PENDING
  window can never be promoted behind a still-chunking request, and a
  request that arrives after the decodes sits BEHIND them in the running
  order (its chunk shrinks to fit instead of trimming their windows). Both
  orders of the original shape (shorts first / long first) are structurally
  unable to fire; the ~900 evidence steps of the two 2026-10-14 runs recorded
  zero trims, consistent with this.
- What the round-7 retune does anyway: raises concurrency (seven long-lived
  shorts with pending windows, topped up between attempts) and calibrates
  the long prompt at runtime onto small budget-leftover targets
  (`s4_leftover_targets` [8, 7, 11] under `s4_max_batched_tokens` 8192),
  measuring the tokenized length from the server's own `usage.prompt_tokens`
  (two fit probes, one candidate measure, word-tail micro-adjustment; each
  attempt rotates a digit in its FIRST sentence so the prefix cache cannot
  absorb the chunk - a fully cached chunk consumes ~no budget). This fires
  if any part of the promotion-order reading above is wrong (e.g. a
  non-FCFS promotion order, a large `draft_slots`, or the mamba
  block-aligned split trimming decode rows), and it is cheap (~25 s) next to
  the boots.
- The two mechanisms that DO reach a trim in this engine, for the record:
  (a) the `max_model_len` clamp (`num_new_tokens = min(..., max_model_len -
  num_computed - num_sampled_tokens_per_step)`) - reachable only with a
  ~262k-token prompt (~90-120 s of prefill per attempt) and with an
  unverified tail-of-request behavior at the clamp boundary; (b) a
  scheduler-side change (trim/echo the armed `num_spec_tokens_to_schedule`
  into the window accounting) - a stage-1b/1c decision, not a workload one.
  Neither is wired by this round.

## What remains manual

1. **Building the stage-1b image** (patch_width_runner.py, round 4 adds
   the C1 counter chain across six files, plus awdbg1b_evidence.py,
   Dockerfile.iter7b + v18b config): the harness runs only after that
   exists; until then every live point reports MISSING_INSTRUMENT. The
   iter7b image tag built on 2026-10-13 predates the round-4 patch (it
   carries the armed wiring but NOT the width-runner/C1 chain) and must be
   rebuilt before any `run`.
2. **The kernel-level stochastic block-verifier gate** (D-009 section b:
   known target/proposal distributions through the actual block verifier,
   rejection/residual/bonus and invalid-tail cases). Needs in-process access
   to the rejection sampler; the HTTP-level D2/D3/D4 checks here are its
   end-to-end shadow, not a substitute. The DETERMINISTIC half of section b's
   closing requirement, the proposal-logit cache alignment checks, is no
   longer manual: round 2 delivered it as P1.A8 / P2.B6/B6b/B6c over the
   C2 `row`/`steps` fields (see C2-alignment); what stays manual is only the
   value-level staleness half (stale q' content), which needs the in-process
   distributions.
3. **EMA-pure armed drive** (P5.E4): a separate boot with an acceptance-cliff
   workload; soft by design (EMA warmup/interval makes the switch timing
   traffic-dependent).
4. **Structured-output / grammar -1 padding path**: `update_draft_token_ids`
   grammar validation and `num_invalid_spec_tokens` are exercised only when a
   guided-decode request runs; the P3 scenarios can be extended with one
   (response_format json_schema) once stage-1b declares structured-output
   support.
5. **Async scheduling coverage**: deliberately excluded (asserted excluded,
   P3.S7) pending the verification D-009 requires.
6. **Shortened-window trigger confirmation (round 7)**: the shape is
   retuned and runtime-calibrated (see "S4 trigger analysis"); whether it
   fires live - or whether the trigger is structurally unreachable in this
   engine build - is settled by the next gate run's S4 detail and
   `res.data["s4_shape"]`. If S4 still WARNs with calibrated attempts in
   range, the scheduler-side options (a trim hook on the armed path, or the
   `max_model_len` clamp shape) are a stage-1b/1c decision, not a workload
   one.
7. **Operator coordination**: every `run` replaces the running server for
   ~2 boots x ~4 min + workloads; schedule accordingly and confirm the
   restore completed (`check-live` afterwards shows the MTP=3 baseline).

## Provenance

- Round 8 (2026-10-15, D-014 conditions 1+2): the C2 field completion and
  the runtime behavioral fingerprints are anchored on the iter7d image
  texts (extracted read-only from qwen38-flash-dgx:iter7d-20261014:
  scheduler.py / gpu/model_runner.py / autoregressive/speculator.py /
  marlin_moe.py); the marlin anchors are the two `moe_wna16_marlin_gemm`
  call sites (`use_atomic_add` hard-set false at :159 and :226, the
  atomic_add=0 proof) and the stable-sort shim block; the fingerprint
  probe reuses the shim's own debug checker as the canonical-order
  asserter; the sampler-side fields read the `SamplingStates` CPU mirrors
  (`temperature.np` / `seeds.np` / `seeds_set`, `gpu/sample/states.py`);
  the verify-side fields read the scheduler's verify accounting in
  `update_from_output` (the same numbers the acc/rej counters fold); the
  autotune-selection walker covers `vllm/third_party/flash_linear_attention`
  and the mamba ops modules plus the `VLLM_TRITON_FORCE_FIRST_CONFIG`
  pinning state (`vllm/triton_utils/force_first_config.py`).
- Gate requirements: D-009 verdict section d (points 1-5) and the two config
  traps (`vllm_adaptive_width.py:96-110`, `:217-264`).
- Counter shapes verified against the live `/metrics` exposition on
  qwen38-flash-dgx:iter7a-20261013 (labels `engine`, `model_name`,
  `position`).
- V2 path identification (D-009's correction of the blueprint): propose call
  at `vllm/v1/worker/gpu/model_runner.py:1893` (`self.speculator.propose`),
  dummy-run call at :784, draft loop
  `gpu/spec_decode/autoregressive/speculator.py:472`
  (`for step in range(1, self.num_speculative_steps)`), fused variant :573,
  first-pass sample into `draft_tokens[:num_reqs, 0]`.
- Scheduler verify-window semantics: `update_from_output`
  (`num_draft_tokens = len(scheduled_spec_token_ids)`), trim
  (`scheduler.py` schedule loop), -1 normalization +
  `num_invalid_spec_tokens` (`update_draft_token_ids_in_output`).
- Controller semantics: `vllm_adaptive_width.py` (modes, oscillate flip on
  observe(), tier merge in `slot_for`), `patch_adaptive_width_armed.py`
  (armed branch, launch_k guard).
- Round-2 alignment contract sources (anchor-excerpts-v2, byte-identical to
  the live iter7a image): `rej-utils-150-300.txt` /
  `rej-utils-330-810-block-likelihood.txt` (draft_logits
  `[max_num_reqs, num_speculative_steps, V]` addressed
  `req_state_idx * stride_0 + draft_step * stride_1`; the
  `draft_step_idx >= num_speculative_steps` bonus skip at :228-232),
  `v2-runner-1880-1935-propose-call.txt` (row ownership:
  `req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens` at
  :1907), `v2-runner-1172-1288-window-logits.txt` (per-token
  `expanded_idx_mapping` / `expanded_local_pos` construction),
  `rej-utils-553-562-invalid-tail.txt` (the -1 tail the valid_len/bounds
  contract leans on).
- Mock-world behavioral model: same phase relations as the live contract -
  round 4 aligns it with the implemented emitter (worker truth: every row
  samples k_eff positions; win == the previous proposal's width with no
  cold-start seeding; the C1 fold lag is tolerated at window edges); the
  selftest proves each D-009 defect class flips the corresponding check
  (see `gate_selftest.py` CASES).
- Round-6 trim-exemption basis: the round-5 re-review's latent caveat (a
  request the token budget trims to 0 tokens is absent from
  `num_scheduled_tokens`, `scheduled_running_reqs` only appends actually
  scheduled requests, scheduler.py:828, while its row can persist in the
  worker batch). For the record: the iter7a engine currently REMOVES
  unscheduled requests from the persistent batch each step
  (`gpu_model_runner._update_states`), so `bs == worker rows` holds
  structurally on the boots the gate runs today; the exemption is a
  bounded safety net for the trim shape (id absent from the paired sched
  line's reqs, win == 0 by absence) that cannot mask a dropped or
  duplicated SCHEDULED row, and its firing count is always reported.
