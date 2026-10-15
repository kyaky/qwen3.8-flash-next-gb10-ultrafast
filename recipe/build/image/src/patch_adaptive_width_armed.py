#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-1c ARMED wiring for the adaptive draft-width controller (P2).

Applied AFTER patch_adaptive_width.py (stage-1a shadow wiring), the anchors
below exist only in the stage-1a-patched scheduler state (image iter7a). Two
anchored edits:

  1. install call gains launch_k=self.num_spec_tokens: the controller
     COORDINATES the candidate ladder with the launch num_speculative_tokens
     (D-009 trap 1, clamp-then-derive, it no longer refuses on
     max(candidate_steps) > launch_k): the configured ladder (default AND
     every tier) is clamped to <= the launch K with a logged warning, a scope
     with nothing left at or below it derives the single-candidate ladder
     [launch_k], and a numeric FORCE above the launch K is still refused (a
     pinned width the launch cannot hold is unsatisfiable), as is an
     oscillating ladder containing the donor 0 candidate (D-009 round 3: the
     square wave would drive the zero width the force contract refuses).
     launch_k is REQUIRED for the driving modes: without it install()
     refuses (the except-arm falls back to static K) because no ladder
     coordination is
     possible, the launch-at-max-width invariant (draft buffers, verify
     graphs and metrics counters are all sized at the launch K).
  2. the dynamic-SD K site (upstream "compute optimal K") gains an elif that
     drives SchedulerOutput.num_spec_tokens_to_schedule from the controller
     whenever it is installed in a driving mode. k_for_step() re-checks the
     mode itself: off/shadow return config K (branch unreachable for them
     anyway), force returns the pinned/oscillating width (G2/G3/G4 harness),
     armed returns the EMA slot width (or the forced square wave under
     FORCE=oscillate).

Transition contract (codex design review task-mv5z1wkw): the width rides
SchedulerOutput and takes effect at the FORWARD boundary of the NEXT proposal
- this step's drafter proposes k_for_step() drafts, and the next step's verify
window absorbs them through request.spec_token_ids (num_tokens_with_spec
grows/shrinks with the previous proposal; grammar-invalidated slots pad with
-1, which rejection sampling treats as never-accepted, exactly). Do NOT clamp
K by the pending width: the pending window self-drains via spec_token_ids and
a pending clamp would pin K at the launch width forever (deadlock at
down-switch is NOT possible through this path; the pending-clamp variant is
the one that deadlocks).

Consumption: vllm/v1/worker/gpu_model_runner.py:5605 already passes
num_spec_tokens_to_schedule into DraftModelProposer.propose, and
SpecDecodeBaseProposer.propose (llm_base_proposer.py:514) accepts and uses it
per call (loop range(self.num_speculative_tokens - 1) at :694). The V2
speculator runner (the deployment's live path) has zero consumers of this
field, that is stage-1b (patch_width_runner.py), NOT this patch.

Untouched by design: dynamic_sd_lookup stays None (the uniform-decode padding
gate requires it, when it is not None the dynamic-SD branch above wins over
this elif), num_prefill_lookahead stays a config quantity (patch_block_drop.py
territory), and the [-1]*launch-K cold-start placeholder keeps its launch
width (all -1s are never-accepted, so a shorter first proposal is exact).

Usage: python3 patch_adaptive_width_armed.py <scheduler.py>
"""
import sys

TARGET = sys.argv[1]
s = open(TARGET).read()

# Guard: this patch must run on the stage-1a state (iter7a), never twice.
assert "_adaptive_width_armed_branch" not in s, "armed wiring already applied"

# 1. install call gains the launch-at-max-width guard argument.
anchor1 = """                self._adaptive_width = _aw_install(
                    _aw_mode,
                    _aw_os.environ.get("VLLM_ADAPTIVE_WIDTH_FORCE"),
                    batch_size_fn=lambda: len(self.running),
                )
"""
new1 = """                self._adaptive_width = _aw_install(
                    _aw_mode,
                    _aw_os.environ.get("VLLM_ADAPTIVE_WIDTH_FORCE"),
                    batch_size_fn=lambda: len(self.running),
                    launch_k=self.num_spec_tokens,
                )
"""
assert s.count(anchor1) == 1, f"anchor1 count {s.count(anchor1)}"
s = s.replace(anchor1, new1)

# 2. the armed branch at the dynamic-SD K site (scheduler.py:1399-1404 in the
#    iter7a image).
anchor2 = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
"""
new2 = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
        # ---- GB10 adaptive width: armed per-step K (P2 stage-1c) ----
        # _adaptive_width_armed_branch
        # dynamic_sd_lookup stays None in this deployment (the uniform-decode
        # padding gate requires it), so this elif is the only per-step K
        # source; if dynamic SD were ever configured it wins over the
        # controller by construction of the elif. k_for_step() re-checks the
        # mode: off/shadow return config K, force returns the pinned or
        # oscillating width (G2/G3/G4), armed returns the EMA slot width.
        # Batch-size basis matches the stage-1a observe() feed
        # (len(self.running)) so both hit the same tier slot.
        elif (
            self._adaptive_width is not None
            and self._adaptive_width.mode in ("armed", "force")
            and len(num_scheduled_tokens) > 0
        ):
            num_spec_tokens_to_schedule = self._adaptive_width.k_for_step(
                len(self.running), self.num_spec_tokens
            )
            if not getattr(self, "_aw_armed_logged", False):
                self._aw_armed_logged = True
                logger.info(
                    "adaptive width armed: driving per-step K "
                    "(mode=%s force=%s k=%d launch_k=%d)",
                    self._adaptive_width.mode,
                    self._adaptive_width.force,
                    num_spec_tokens_to_schedule,
                    self.num_spec_tokens,
                )
        # ---- end GB10 adaptive width armed branch ----
"""
assert s.count(anchor2) == 1, f"anchor2 count {s.count(anchor2)}"
s = s.replace(anchor2, new2)

open(TARGET, "w").write(s)
import ast
ast.parse(open(TARGET).read())
print("adaptive width armed wiring applied OK")
