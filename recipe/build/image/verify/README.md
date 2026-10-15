---
author: glm-5.3-flash
date: 2026-10-14
task: PLAN.md#P2 (stage-1b round-4 fix list, item 5: provenance)
type: field-report
status: draft
reviewed_by: pending-codex
evidence: recipe/build/image/verify/ (this directory; canonical copy committed to the fork -
  the 09196bb precedent: process artifacts are archived with the recipe they verify);
  agent-workspace mirror at agents/glm-5.3-flash-integrator/runs/2026-10-14-p2-stage1b-verify/
---

# Stage-1b chain verification (round 4)

Reproducible proof that the committed patch chain produces exactly the
state it claims, starting from the pinned live image. The round-4 review
found the previous implementation's byte-identical claim unauditable
(`recipe/build/image/src/_stage1b_verify/` was empty); this directory is
that evidence, regenerated for the round-4 patch.

## What was run (2026-10-14, read-only wrt the server)

1. The six pristine inputs were extracted from the pinned image
   `qwen38-flash-dgx:iter7a-20261013` via `docker create` + `docker cp` +
   `docker rm`, NO container was started, no build, no restart, no
   benchmark. Cross-check: all six md5s are byte-identical to read-only
   `docker exec` reads of the RUNNING `qwen38-flash` container (which is
   iter7a-20261013), so image state == live state.
2. The chain was applied to those copies in order:
   `patch_adaptive_width_armed.py` (stage-1c, scheduler.py) →
   `patch_width_runner.py` (stage-1b round 4: scheduler.py,
   gpu_model_runner.py, speculator.py, metrics.py, outputs.py,
   spec_decode/utils.py). Every anchor matched exactly once, every file
   passed `ast.parse` (see `chain.log`).
3. Negative controls (all refuse, see `chain.log`):
   - double application of the armed patch (`armed wiring already applied`);
   - double application of the width runner (`found '_awdbg_sched_emitter'`);
   - out-of-order application (width runner on an un-armed scheduler:
     `required chain marker missing: '_adaptive_width_armed_branch'`);
   - the pristine iter6e scheduler (stage-1a shadow only) refuses the
     armed patch (`anchor1 count 0`) and the width runner (chain marker).

## Files

- `md5-pristine.txt` / `md5-patched.txt`, input/output manifests.
- `iter7a-to-stage1b-round4.diff`, the full unified diff the chain
  produces (609 lines across the six files; reviewable without docker).
- `chain.log`, the patcher run log including the negative controls.

## Scope note (honest)

These artifacts prove the PATCHER's output on the pinned iter7a inputs.
The currently built `qwen38-flash-dgx:iter7b-20261013` image predates the
round-4 patch (verified: its scheduler.py carries the armed branch marker,
count 1, and ZERO width-runner/C1 markers) and must be REBUILT from this
recipe before any gate `run`; the rebuild's in-image state is then
checksum-verifiable against `md5-patched.txt` (same chain, same inputs).

## Round 5 addendum (2026-10-14, gate-run 2026-10-14-014411 fixes)

The round-5 patch changes (`patch_width_runner.py`: the worker emitter gated
to SCHEDULED batches, the C1 position-0 count moved to the prefill rejoin,
the C1 `ModelRunnerOutput.sampled_per_pos` carrier + the sync-channel
liveness log line) were re-verified the same way, offline:

1. The chain was applied to fresh copies of the same six pristine inputs
   (`md5-pristine.txt`, byte-identical to the pinned iter7a image): armed
   wiring -> round-5 width runner. Every anchor matched exactly once, every
   file passed `ast.parse`.
2. Outputs: `md5-patched-round5.txt`; the full unified diff is
   `iter7a-to-stage1b-round5.diff` (718 lines across the six files).
   `metrics.py` and `spec_decode_utils.py` are byte-identical to the
   round-4 state (this round touches four of the six files).
3. Cross-check: the previously built iter7b image still carries the round-4
   state; the NEXT rebuild must produce exactly `md5-patched-round5.txt`
   (same chain, same inputs).
4. Negative controls unchanged (double-application and out-of-order guards
   still refuse; the round-5 markers extend the guard lists).
