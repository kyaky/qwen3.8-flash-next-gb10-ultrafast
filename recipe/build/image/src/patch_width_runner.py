#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P2 stage-1b RUNNER-side patch: per-call draft width + C1/C2 evidence.

Applied ON TOP of the iter7b image state (stage-1a shadow wiring + armed
scheduler wiring are already in scheduler.py; gpu/model_runner.py,
autoregressive/speculator.py, spec_decode/metrics.py, v1/outputs.py and
spec_decode/utils.py are still pristine upstream text -- the 1a/1c patches
touch only scheduler.py, so the chain order is:

    iter6e pristine -> patch_adaptive_width.py (stage-1a, scheduler)
                     -> patch_adaptive_width_armed.py (stage-1c, scheduler)
                     -> patch_width_runner.py (stage-1b, six files)

Round 5 (gate-run 2026-10-14-014411 fixes; codex alignment task-mv6tvlfv):

  - WORKER EMITTER GATED TO SCHEDULED BATCHES: the runner-side JIT kernel
    warmup (gpu/warmup.py warmup_kernels) runs 6 real non-dummy steps
    (prefill + 5 decode steps, rows 8/8/2/2/1/1) whose fabricated
    SchedulerOutputs carry the make_empty default num_spec_tokens_to_
    schedule=0. The round-4 emitter keyed on `is not None` fired on all
    six, the scheduler never paired them, and the worker seq counter ran
    6 ahead of the sched seq counter for the whole boot (the gate's
    A8b/B4a/B4/E2 desyncs). The emitter (and the C2 seq counters) now
    fire only when requested_k > 0 -- real scheduled steps always ship
    >= 1 on a spec boot -- so worker:sched is 1:1 and both seq counters
    start together at the first scheduled step (C2 pairing invariant).
  - POSITION-0 ACCUMULATION MOVED TO THE PREFILL REJOIN: the executed-sample
    counter's position-0 count now happens after the eager/captured-graph
    prefill branches rejoin (num_reqs once per real call, padding
    excluded), recording completed work; the round-4 entry-side count
    pre-dated the prefill branches.
  - C1 TRANSPORT FIXED FOR THE ASYNC SCHEDULER: SchedulerConfig.
    async_scheduling=None AUTO-resolves to True for MTP boots ("mtp" is an
    EagleModelTypes member), so the gate boots silently ran the
    AsyncScheduler and the engine core never called update_draft_token_ids
    -- the round-4 chain's only C1 staging site was dead and the counter
    stayed at a zero delta over 23 steps / 249 evidence samples. The same
    step's list now ALSO rides ModelRunnerOutput.sampled_per_pos and is
    staged by update_from_output AFTER that step's fold (the one-step lag
    and the single fold per list are preserved on BOTH scheduler modes);
    update_draft_token_ids additionally logs a one-time sync-channel
    liveness line the gate's P3.S7 asserts as engine truth.

Anchored, assert-guarded edits (every anchor must match exactly once; every
edited file is ast.parse-checked afterwards):

  scheduler.py (vllm/v1/core/sched/scheduler.py)
    - per-request accepted/rejected collection inside update_from_output,
    - the C2 SCHED evidence line at the END of update_from_output (the C2
      contract pins acc/rej to "already computed in update_from_output", so
      the emitter lives there rather than at the armed branch site; the
      armed branch is guarded to exist via its marker comment),
    - C1 staging: update_draft_token_ids stages the worker's per-position
      executed-sample counts (DraftTokenIds.sampled_per_pos) and the end of
      update_from_output folds the staged list into SpecDecodingStats. The
      post-step channel arrives AFTER the step's stats have been sealed, so
      the fold lands one engine step later -- exactly the verify step of the
      proposal that produced the samples (README phase relation 4).

  gpu/model_runner.py (vllm/v1/worker/gpu/model_runner.py -- the V2 runner)
    - ExecuteModelState carries num_spec_tokens_to_schedule (None on
      dummy/profile runs) from execute_model into sample_tokens,
    - the V2 propose call site passes it into speculator.propose,
    - the proposal storage write narrows to the returned k_eff columns,
    - DraftTokensHandler receives the k_eff-wide proposal AND the
      speculator's executed-sample counter, so the NEXT verify window keeps
      this proposal's own length (the three-quantity separation: launch
      capacity / current verification lengths / next-proposal K -- D-009
      section a.1) and the C1 counter rides the same per-step channel,
    - (round 5) the C1 counts also ride ModelRunnerOutput.sampled_per_pos
      -- the carrier the async scheduler can see (it never pulls the
      DraftTokenIds channel),
    - the C2 WORKER evidence line after each propose that belongs to a
      SCHEDULED batch (requested_k > 0; the JIT-kernel-warmup steps carry
      the make_empty default 0 and never reach the scheduler), reporting
      WORKER TRUTH (round 4): every batch row executes one draft sample at
      every position 0..k_eff-1 on this drafter, so valid_len == k_eff and
      steps == range(k_eff) on EVERY row; sampled_per_pos is the
      speculator's own counter, not a function of scheduler-visible
      windows.

  autoregressive/speculator.py (vllm/v1/worker/gpu/spec_decode/...)
    - propose() accepts num_spec_tokens_to_schedule; per-call width
      k_eff = min(scheduled K, buffered capacity); None/0 fall back to the
      launch config K (dummy/profile/pre-armed boots stay byte-identical),
    - the draft decode loop bound honors k_eff (both the non-fused and the
      fused variant), the returned draft_tokens slice honors k_eff,
    - the draft_logits cache keeps its K_max-strided buffer (columns
      >= k_eff unused -- the masking contract covers them);
      update_draft_inputs is untouched (it already takes a runtime step +
      stride; the loop bound is what changed),
    - C1: the per-position executed-sample counter. Reset per real propose
      (staged None on dummy/profile so boot prewarm cannot pollute it and a
      stale list is never re-shipped); position 0 counted at the PREFILL
      REJOIN (round 5: after the eager/captured-graph prefill branches
      rejoin, num_reqs once per real call -- the first-pass sample executes
      for every batch row by then, eagerly in _prefill or inside the
      captured prefill graph, either way exactly one sample per row;
      graph padding excluded); positions 1..k_eff-1 counted
      after each eager _generate_draft call in both loop variants. FULL-graph
      decode replay would bake the samples into the captured graph and is
      NOT counted (the same physical limit class as the baked loop bound;
      this deployment runs eager draft decode -- PIECEWISE target graphs,
      fused draft decode disabled for the QSA backend).

  spec_decode/metrics.py (vllm/v1/spec_decode/metrics.py) -- C1 frontend
    - SpecDecodingStats carries num_sampled_draft_tokens_per_pos
      (initialized [0]*num_spec_tokens in new(), same sizing rule as the
      accepted/draft per-pos vectors),
    - SpecDecodingProm creates
      vllm:spec_decode_num_sampled_draft_tokens_per_pos next to the
      accepted-per-pos counter (same label family engine/model_name plus
      position, pre-created for range(num_spec_tokens) at init; empty for
      diffusion boots, where observe() no-ops on the empty dict),
    - SpecDecodingProm.observe increments it per position (an empty
      per-position vector increments nothing -- never an IndexError).

  outputs.py (vllm/v1/outputs.py) -- C1 transport
    - DraftTokenIds gains sampled_per_pos: list[int] | None = None (the
      worker's per-position executed-sample counts for the step that
      produced the proposal; None when the step ran no real propose).
    - (round 5) ModelRunnerOutput gains the same sampled_per_pos carrier:
      the scheduler's fold stages from it in update_from_output, which is
      the only staging site the ASYNC scheduler feeds (it never pulls the
      post-step DraftTokenIds channel).

  spec_decode/utils.py (vllm/v1/worker/gpu/spec_decode/utils.py) -- C1
    - DraftTokensHandler stages sampled_per_pos in set_draft_tokens and
      ships it in get_draft_tokens on BOTH paths -- the structured-output
      copy path AND the no-structured-output placeholder path
      ([[-1]*width] rows), which is the path every gate boot takes.

Masking / lifetime contract (D-009 section c, made explicit): a k_eff-wide
proposal leaves storage columns >= k_eff and draft-logit columns >= k_eff
with their previous content; they are NEVER read, because
  (a) the verify window length for each row equals the proposal width the
      DraftTokensHandler carried (or a budget trim of it), and
  (b) the verify/aggregate kernels mask by the scheduled window length and
      by accepted_length (rej-utils :627 stores unconditionally, :667 masks),
and on a width GROW the loop re-executes every position 0..k_eff-1,
overwriting each newly valid column before exposure (P2.B6b). The
[-1] tail (draft_sampled >= 0, rej-utils :553-562) stays never-accepted, so
trailing placeholders are distribution-exact.

Physical limit documented (not hit by this deployment): the FUSED draft
decode loop under FULL cudagraph replay bakes its loop bound at capture
time. This deployment runs PIECEWISE target graphs with the fused path
disabled (QSA backend lacks draft-decode metadata update), so both loop
variants execute eagerly and honor k_eff at runtime; per-width capture is
stage-3 territory.

Usage:
    python3 patch_width_runner.py <scheduler.py> <gpu_model_runner.py> \
        <autoregressive_speculator.py> <spec_decode_metrics.py> \
        <outputs.py> <spec_decode_utils.py>
"""
import ast
import sys

if len(sys.argv) != 7:
    print(
        "usage: patch_width_runner.py <scheduler.py> <gpu_model_runner.py> "
        "<autoregressive_speculator.py> <spec_decode_metrics.py> "
        "<outputs.py> <spec_decode_utils.py>",
        file=sys.stderr,
    )
    sys.exit(64)

SCHED_PATH = sys.argv[1]
RUNNER_PATH = sys.argv[2]
SPEC_PATH = sys.argv[3]
METRICS_PATH = sys.argv[4]
OUTPUTS_PATH = sys.argv[5]
SDUTILS_PATH = sys.argv[6]


def patch(path, edits, must_contain=(), must_not_contain=()):
    s = open(path).read()
    for marker in must_contain:
        assert marker in s, f"{path}: required chain marker missing: {marker!r}"
    for marker in must_not_contain:
        assert marker not in s, (
            f"{path}: patch already applied (found {marker!r})"
        )
    for i, (anchor, replacement) in enumerate(edits):
        n = s.count(anchor)
        assert n == 1, f"{path}: anchor#{i} count {n} (want 1): {anchor[:80]!r}"
        s = s.replace(anchor, replacement)
    open(path, "w").write(s)
    ast.parse(open(path).read())
    print(f"patch_width_runner: {path} OK ({len(edits)} edits, ast.parse pass)")


# ===========================================================================
# 1. scheduler.py -- acc/rej collection, the C2 SCHED evidence line, and the
#    C1 staged fold of the worker's executed-sample counts
# ===========================================================================
patch(
    SCHED_PATH,
    edits=[
        # ---- SC1: evidence plumbing at the top of update_from_output ----
        (
            """        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
""",
            """        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        # ---- GB10 AWDBG1B: sched evidence plumbing (P2 stage-1b) ----
        # _awdbg_sched_evidence
        # Lazy helper import (the module sits next to vllm in site-packages,
        # copied by the stage-1b image build); evidence_enabled() caches the
        # VLLM_SPECDBG_EVIDENCE read, so with the gate off the per-step cost
        # is one sys.modules lookup.
        from awdbg1b_evidence import (  # noqa: E402 - evidence-gated helper
            evidence_enabled as awdbg_ev_enabled,
        )
        # Per-request (accepted, rejected) draft counts for this step's
        # sched evidence line; requests without a spec verify stay (0, 0).
        awdbg_acc: dict[str, tuple[int, int]] = {}
        # ---- end GB10 AWDBG1B sched evidence plumbing ----
""",
        ),
        # ---- SC2: collect acc/rej next to the stage-1a shadow observation --
        (
            """                # ---- end GB10 adaptive width ----
""",
            """                # ---- end GB10 adaptive width ----
                # ---- GB10 AWDBG1B: per-request verify counts (P2 stage-1b)
                # ---- The C2 sched line reports the accepted/rejected counts
                # computed HERE (update_from_output); num_rejected counts
                # draft slots that did not accept (num_draft - num_accepted).
                if awdbg_ev_enabled():
                    awdbg_acc[req_id] = (int(num_accepted), int(num_rejected))
                # ---- end GB10 AWDBG1B verify counts ----
""",
        ),
        # ---- SC3: the C2 SCHED evidence line at the end of the method ----
        (
            """            eco.scheduler_stats = stats

        return engine_core_outputs
""",
            """            eco.scheduler_stats = stats

        # ---- GB10 AWDBG1B: sched evidence line (P2 stage-1b) ----
        # _awdbg_sched_emitter
        # One line per engine step that scheduled a batch, emitted at the END
        # of update_from_output where scheduler_output AND this step's
        # per-request acc/rej counts are both in scope (the C2 contract pins
        # acc/rej to update_from_output; empty steps emit nothing, matching
        # the worker side, which only runs for non-empty batches -- the two
        # seq counters then pair 1:1). requested_k is the post-armed-branch
        # value shipped in THIS step's SchedulerOutput; win is the pending
        # verify window (one-step lag); reqs order is the scheduler's
        # scheduled order (num_scheduled_tokens insertion order, NOT
        # id-sorted -- the C2 order contract).
        if num_scheduled_tokens and awdbg_ev_enabled():
            from awdbg1b_evidence import (  # noqa: E402 - evidence-gated
                build_sched_record as _awdbg_bsr,
                emit_line as _awdbg_emit,
            )
            _awdbg_emit(
                logger,
                _awdbg_bsr(
                    launch_k=self.num_spec_tokens,
                    requested_k=scheduler_output.num_spec_tokens_to_schedule,
                    scheduled_req_ids=list(num_scheduled_tokens),
                    spec_windows=scheduler_output.scheduled_spec_decode_tokens,
                    acc_rej=awdbg_acc,
                    mode=(
                        str(self._adaptive_width.mode)
                        if self._adaptive_width is not None
                        else "off"
                    ),
                ),
            )
        # ---- end GB10 AWDBG1B sched evidence ----
        return engine_core_outputs
""",
        ),
        # ---- SC4: C1 fold -- the staged worker list lands in THIS step's
        # SpecDecodingStats, one step after the samples executed (the
        # post-step DraftTokenIds channel arrives after the previous
        # step's stats were sealed; that step is the proposal's verify
        # step, so samples and the drafts/accepted counters of the same
        # stats object describe the same proposals). Round 5 adds the
        # ModelRunnerOutput staging below: the round-4 gate run left the
        # C1 counter at a ZERO delta over 23 steps / 249 evidence samples
        # because SchedulerConfig.async_scheduling=None AUTO-resolves to
        # True for MTP boots ("mtp" is an EagleModelTypes member and the
        # executor supports async), and the engine core then NEVER calls
        # update_draft_token_ids -- the round-4 chain's only staging site
        # was dead. The same step's list therefore ALSO rides
        # ModelRunnerOutput.sampled_per_pos and is staged here. ----
        (
            """        if (
            stats := self.make_stats(
                spec_decoding_stats,
""",
            """        # ---- GB10 C1: fold the worker's executed-sample counts (P2
        # stage-1b) ---- _awdbg_c1_fold
        # The per-position sample counts travel on the post-step
        # DraftTokenIds channel (sampled_per_pos), which the engine core
        # pulls AFTER this method has already sealed the previous step's
        # SpecDecodingStats (update_draft_token_ids runs in post_step).
        # The staged list therefore folds into THIS step's stats one engine
        # step after the samples executed -- which is exactly the verify
        # step of the proposal that produced them (the one-step lag), so
        # the sampled counter and the drafts/accepted counters of the same
        # stats object describe the same proposals. Window-level gate
        # checks tolerate the residual boundary skew (README phase
        # relation 4). None stages nothing (dummy/profile propose,
        # speculator-less boots).
        awdbg_pending = getattr(self, "_awdbg_pending_samples", None)
        self._awdbg_pending_samples = None
        if awdbg_pending and self.log_stats:
            if spec_decoding_stats is None:
                spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
            for awdbg_i, awdbg_v in enumerate(
                awdbg_pending[: self.num_spec_tokens]
            ):
                spec_decoding_stats.num_sampled_draft_tokens_per_pos[
                    awdbg_i
                ] += int(awdbg_v)
        # ---- GB10 C1 (round 5): stage the counts from the ModelRunnerOutput
        # carrier ---- _awdbg_c1_stage_out
        # The async scheduler never pulls the post-step DraftTokenIds channel
        # (the engine core skips update_draft_token_ids whenever
        # scheduler_config.async_scheduling is set), so the SAME step's list
        # also rides ModelRunnerOutput.sampled_per_pos (the V2 runner sets it
        # right after propose). It is staged HERE -- AFTER this step's fold
        # drained the previous step's counts -- so the one-step lag (README
        # phase relation 4) is preserved on BOTH scheduler modes and each
        # step's list is folded exactly once. In sync mode the post-step
        # DraftTokenIds staging (update_draft_token_ids) re-stages the same
        # list afterwards -- identical value, no double count. None keeps the
        # slot cleared by the fold above (no real propose this step).
        awdbg_out_samples = getattr(model_runner_output, "sampled_per_pos", None)
        if awdbg_out_samples is not None:
            self._awdbg_pending_samples = [int(v) for v in awdbg_out_samples]
        if (
            stats := self.make_stats(
                spec_decoding_stats,
""",
        ),
        # ---- SC5: C1 staging in update_draft_token_ids (the sync path's
        # post-step consumer of DraftTokenIds). Round 5 adds the one-time
        # sync-channel liveness log line: the gate's P3.S7 greps it as
        # ENGINE TRUTH that the sync scheduler is actually running -- the
        # round-4 gate run reported "async flags: none" from an argv-only
        # grep while SchedulerConfig.async_scheduling had auto-resolved to
        # True for the MTP boot and this method never executed. ----
        (
            """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
""",
            """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        # ---- GB10 C1: stage the worker's executed-sample counts (P2
        # stage-1b) ---- _awdbg_c1_stage
        # sampled_per_pos rides DraftTokenIds (None on a dummy/profile
        # propose or a speculator-less boot). It is staged here and folded
        # into the NEXT update_from_output's SpecDecodingStats (the fold
        # site at the end of that method) -- post_step runs after the
        # step's stats have already been sealed.
        self._awdbg_pending_samples = draft_token_ids.sampled_per_pos
        # ---- GB10 stage-1b (round 5): one-time sync-channel liveness line
        # ---- _awdbg_sync_channel_log
        # This method only runs when the engine core pulls the post-step
        # DraftTokenIds channel -- i.e. when scheduler_config.
        # async_scheduling is FALSE. MTP boots AUTO-enable async scheduling
        # (async_scheduling=None resolves to True for the "mtp" method), which
        # silently starved this channel -- and with it the C1 fold -- in the
        # round-4 gate run while every argv looked clean. One line per
        # scheduler process, ungated (cheap; production-visible).
        if not getattr(self, "_awdbg_sync_channel_logged", False):
            self._awdbg_sync_channel_logged = True
            logger.info(
                "GB10 stage-1b: sync draft-token channel live "
                "(update_draft_token_ids runs; async scheduling off)"
            )
        for req_id, spec_token_ids in zip(
""",
        ),
    ],
    # Chain guard: the stage-1c armed branch must already be present (this
    # patch is built and gated on top of the iter7b image state).
    must_contain=["_adaptive_width_armed_branch"],
    must_not_contain=[
        "_awdbg_sched_emitter",
        "_awdbg_sched_evidence",
        "_awdbg_c1_stage",
        "_awdbg_c1_fold",
        "_awdbg_c1_stage_out",
        "_awdbg_sync_channel_log",
    ],
)

# ===========================================================================
# 2. gpu/model_runner.py (V2) -- kwarg threading, narrowed storage write,
#    proposal-width handler + C1 transport, and the C2 WORKER evidence line
# ===========================================================================
patch(
    RUNNER_PATH,
    edits=[
        # ---- MR1: ExecuteModelState carries the per-step proposal K ----
        (
            """class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    finished_req_ids: set[str]
    ec_connector_output: ECConnectorOutput | None
    routed_experts: RoutedExpertsTensors | None
""",
            """class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    finished_req_ids: set[str]
    ec_connector_output: ECConnectorOutput | None
    routed_experts: RoutedExpertsTensors | None
    # ---- GB10 adaptive width (P2 stage-1b) ---- the scheduler's
    # next-proposal K for THIS step (SchedulerOutput.num_spec_tokens_to_
    # schedule, post-armed-branch value). None on dummy/profile runs; the
    # worker evidence emitter fires only when it is a POSITIVE int (round 5:
    # the runner-side JIT kernel warmup fabricates SchedulerOutputs whose
    # num_spec_tokens_to_schedule stays at the make_empty default 0 -- those
    # warmup steps never reach the scheduler, so they must not emit worker
    # lines or the two C2 seq counters desync for the whole boot).
    num_spec_tokens_to_schedule: int | None = None
""",
        ),
        # ---- MR2: populate it at the execute_model construction site ----
        (
            """        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
            ec_connector_output=ec_connector_output,
            routed_experts=routed_experts,
        )
""",
            """        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
            ec_connector_output=ec_connector_output,
            routed_experts=routed_experts,
            # ---- GB10 adaptive width (P2 stage-1b) ---- None marks the
            # dummy/profile path so the drafter keeps the launch width. The
            # evidence emitter (MR4) additionally requires a POSITIVE value:
            # the JIT kernel warmup (gpu/warmup.py warmup_kernels) runs real
            # non-dummy steps whose fabricated SchedulerOutput carries the
            # make_empty default num_spec_tokens_to_schedule=0 -- real
            # scheduled steps always ship >= 1 on a spec boot (schedule()
            # seeds the field with num_spec_tokens before the armed branch).
            num_spec_tokens_to_schedule=(
                None if dummy_run else scheduler_output.num_spec_tokens_to_schedule
            ),
        )
""",
        ),
        # ---- MR3: unpack it in sample_tokens (before the state is dropped)
        (
            """        routed_experts = self.execute_model_state.routed_experts
        self.execute_model_state = None
""",
            """        routed_experts = self.execute_model_state.routed_experts
        # ---- GB10 adaptive width (P2 stage-1b) ---- the per-step proposal
        # K (None on dummy/profile runs).
        awdbg_requested_k = self.execute_model_state.num_spec_tokens_to_schedule
        self.execute_model_state = None
""",
        ),
        # ---- MR4: the propose call gains the kwarg; the storage write
        # narrows to the returned k_eff columns; the C2 WORKER evidence line
        # fires after each non-dummy propose. ----
        (
            """            with use_workspace_lane(self._draft_workspace_lane):
                draft_tokens = self.speculator.propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings_by_layer,
                    spec_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    self.req_states.last_sampled_tokens,
                    self.req_states.next_prefill_tokens,
                    self.sampler.sampling_states.temperature.gpu,
                    self.sampler.sampling_states.seeds.gpu,
                    mm_inputs=mm_inputs,
                )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            if self.adaptive_verification is not None:
                self.adaptive_verification.record_confidences(
                    self.speculator.draft_token_confidence_probs, input_batch
                )
""",
            """            with use_workspace_lane(self._draft_workspace_lane):
                draft_tokens = self.speculator.propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings_by_layer,
                    spec_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    self.req_states.last_sampled_tokens,
                    self.req_states.next_prefill_tokens,
                    self.sampler.sampling_states.temperature.gpu,
                    self.sampler.sampling_states.seeds.gpu,
                    mm_inputs=mm_inputs,
                    # ---- GB10 adaptive width: per-call draft width (P2
                    # stage-1b) ---- next-proposal K from the armed branch;
                    # None on dummy/profile runs (the speculator falls back
                    # to the launch config K there).
                    num_spec_tokens_to_schedule=awdbg_requested_k,
                )
            # ---- GB10 adaptive width: narrowed proposal storage (P2
            # stage-1b) ---- the proposal is k_eff = draft_tokens.shape[1]
            # columns wide (k_eff <= launch K); only those columns are fresh
            # this step. Columns >= k_eff keep their previous content and
            # are NEVER read: each row's verify window length comes from the
            # scheduler's scheduled windows (<= the stored proposal's width)
            # and the verify/aggregate kernels mask by that length
            # (rej-utils :627 stores unconditionally, :667 masks by
            # accepted_length) -- the explicit lifetime/masking contract
            # D-009 section c asks for.
            self.req_states.draft_tokens[
                input_batch.idx_mapping, : draft_tokens.shape[1]
            ] = draft_tokens
            if self.adaptive_verification is not None:
                self.adaptive_verification.record_confidences(
                    self.speculator.draft_token_confidence_probs, input_batch
                )
            # ---- GB10 AWDBG1B: worker evidence line (P2 stage-1b) ----
            # _awdbg_worker_emitter
            # One line per propose that belongs to a SCHEDULED batch: the
            # gate is awdbg_requested_k being a POSITIVE int (round 5). The
            # round-4 gate run emitted one line per NON-DUMMY propose and
            # thereby six boot-time JIT-kernel-warmup proposes
            # (gpu/warmup.py warmup_kernels runs prefill + 5 decode steps
            # through real execute_model calls with fabricated
            # SchedulerOutputs whose num_spec_tokens_to_schedule stays at
            # the make_empty default 0) that the scheduler never paired --
            # the worker seq counter ran 6 ahead of the sched seq counter
            # for the whole boot and every seq-join in the gate desynced.
            # Real scheduled steps always carry requested_k >= 1 (schedule()
            # seeds the field with num_spec_tokens before the armed branch),
            # so this gate makes worker:sched exactly 1:1 and the two
            # per-source seq counters start together at the first scheduled
            # step (the C2 pairing invariant). WORKER TRUTH (round 4): every
            # batch row executes one draft sample at every position
            # 0..k_eff-1 on this drafter (first-pass sample + each
            # _generate_draft call), so valid_len == k_eff and
            # steps == range(k_eff) on EVERY row -- rows without a scheduled
            # verify window (chunked prefill) consume their samples too and
            # simply carry empty windows scheduler-side. sampled_per_pos is
            # the speculator's OWN executed-sample counter for this step
            # (the SAME list that feeds the C1 Prometheus counter via
            # DraftTokenIds.sampled_per_pos and ModelRunnerOutput.
            # sampled_per_pos). reqs lists EVERY batch row in input_batch
            # order with its req_state row. Env gate:
            # VLLM_SPECDBG_EVIDENCE=1.
            if awdbg_requested_k is not None and awdbg_requested_k > 0:
                from awdbg1b_evidence import (  # noqa: E402 - evidence-gated
                    build_worker_record as _awdbg_bwr,
                    emit_line as _awdbg_emit,
                    evidence_enabled as _awdbg_ev,
                )
                if _awdbg_ev():
                    _awdbg_emit(
                        logger,
                        _awdbg_bwr(
                            k_eff=int(draft_tokens.shape[1]),
                            launch_k=self.num_speculative_steps,
                            sampled_per_pos=self.speculator.awdbg_sampled_per_pos,
                            req_ids=input_batch.req_ids,
                            idx_rows=input_batch.idx_mapping.tolist(),
                        ),
                    )
            # ---- end GB10 AWDBG1B worker evidence ----
""",
        ),
        # ---- MR5: the handler carries the PROPOSAL width, so the NEXT
        # verify window keeps this proposal's own length; the C1 counter
        # rides the same channel. The speculator-None case (diffusion LLMs
        # reuse the draft-token path without a speculator) is guarded: the
        # expression falls back to the always-defined req_states buffer, so
        # a speculator-None + num_speculative_steps>0 boot keeps the exact
        # upstream behavior instead of raising NameError. ----
        (
            """            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )
""",
            """            # ---- GB10 adaptive width: proposal-width carrier (P2
            # stage-1b) ---- in the sync path the scheduler's
            # request.spec_token_ids become [-1]*width placeholder rows of
            # exactly THIS width (DraftTokensHandler), so the next verify
            # window keeps the proposal's own length (one-step lag) while
            # the armed branch already drives the NEXT proposal's K -- the
            # three quantities stay separate (D-009 a.1: launch capacity /
            # current verification lengths / next-proposal K). The
            # speculator-None case (diffusion LLMs) falls back to the
            # always-defined req_states buffer: those boots have no armed K
            # and keep the full launch width, exactly as upstream.
            # ---- GB10 C1 (P2 stage-1b): the executed-sample counter rides
            # the same per-step channel (DraftTokenIds.sampled_per_pos);
            # None on a dummy/profile propose (the speculator stages None
            # there) and on speculator-less (diffusion) boots.
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                draft_tokens if self.speculator is not None
                else self.req_states.draft_tokens[input_batch.idx_mapping],
                sampled_per_pos=(
                    self.speculator.awdbg_sampled_per_pos
                    if self.speculator is not None
                    else None
                ),
            )
            # ---- GB10 C1 (round 5): the executed-sample counts also ride
            # the ModelRunnerOutput ---- _awdbg_c1_carrier
            # The async scheduler never pulls the post-step DraftTokenIds
            # channel (the engine core skips update_draft_token_ids whenever
            # scheduler_config.async_scheduling is set -- and MTP boots
            # AUTO-enable it: async_scheduling=None resolves to True for the
            # "mtp" method), so the round-4 chain's only staging site was
            # dead and the C1 counter stayed at zero. The scheduler-side
            # fold therefore ALSO stages from this field (update_from_output
            # reads model_runner_output directly, on both scheduler modes).
            # Both carriers hold the SAME step's list; the fold drains the
            # staging slot exactly once per step, so there is no double
            # count. model_runner_output was constructed above; assigning
            # the field here is safe (AsyncOutput.get_output only overwrites
            # the other fields).
            model_runner_output.sampled_per_pos = (
                self.speculator.awdbg_sampled_per_pos
                if self.speculator is not None
                else None
            )
""",
        ),
    ],
    must_not_contain=[
        "_awdbg_worker_emitter",
        "num_spec_tokens_to_schedule=",
        "_awdbg_c1_carrier",
    ],
)

# ===========================================================================
# 3. autoregressive/speculator.py -- per-call width k_eff + the C1 counter
# ===========================================================================
patch(
    SPEC_PATH,
    edits=[
        # ---- SP1: the active loop-bound attribute + the C1 counter slot ----
        (
            """        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False
""",
            """        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False
        # ---- GB10 adaptive width: active per-call draft width (P2 stage-1b)
        # ---- _aw_active_steps is the decode-loop bound of the CURRENT
        # propose call (min(scheduled K, buffered capacity)). propose() sets
        # it on entry and restores the launch width before returning, so
        # capture/dummy paths (which never run inside propose) always see
        # the full launch width.
        self._aw_active_steps = self.num_speculative_steps
        # ---- GB10 C1 (P2 stage-1b): per-position executed-sample counter.
        # Reset to a fresh zero list at every REAL propose entry and staged
        # to None on dummy/profile calls, so a stale list is never
        # re-shipped and boot prewarm cannot pollute the counter; the
        # runner reads it right after propose returns and ships it on
        # DraftTokenIds.sampled_per_pos (C1 transport) and in the worker
        # evidence line (C2).
        self.awdbg_sampled_per_pos: list[int] | None = None
""",
        ),
        # ---- SP2: the propose signature gains the optional kwarg ----
        (
            """        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
""",
            """        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        # ---- GB10 adaptive width: per-call draft width (P2 stage-1b) ----
        # The scheduler's next-proposal K for THIS step (armed branch).
        # None/0/dummy runs fall back to the launch config K; the value is
        # clamped to the buffered capacity (launch-at-max-width invariant:
        # buffers, cudagraphs and the rejection sampler stay launch-K sized).
        num_spec_tokens_to_schedule: int | None = None,
    ) -> torch.Tensor:
""",
        ),
        # ---- SP3: derive k_eff at propose entry + stage the C1 counter ----
        (
            """        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
""",
            """        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        # ---- GB10 adaptive width: per-call draft width (P2 stage-1b) ----
        # _aw_per_call_k
        # k_eff = min(schedule K, buffered capacity); None/0 -> launch K, so
        # dummy/profile/pre-armed boots keep the exact launch behavior
        # (byte-identical off floor). The armed branch never ships 0 (the
        # controller refuses a zero-width ladder candidate).
        k_eff = self.num_speculative_steps
        if num_spec_tokens_to_schedule is not None:
            awdbg_requested = int(num_spec_tokens_to_schedule)
            if awdbg_requested >= 1:
                k_eff = min(awdbg_requested, self.num_speculative_steps)
        self._aw_active_steps = k_eff
        # ---- end GB10 adaptive width per-call K ----
        # ---- GB10 C1: per-position executed-sample counter (P2 stage-1b)
        # ---- _awdbg_c1_counter
        # A real propose records the samples it is about to execute;
        # dummy/profile runs stage None (no stale list is ever re-shipped,
        # and boot prewarm cannot pollute the counter). Round 5: position 0
        # is counted at the PREFILL REJOIN below (_awdbg_c1_pos0) -- after
        # the eager/captured-graph prefill branches rejoin -- so the counter
        # records COMPLETED work once per real call; the round-4 entry-side
        # count pre-dated the prefill branches. Positions 1..k_eff-1 are
        # counted after each eager _generate_draft call (both loop variants
        # below).
        if dummy_run or is_profile:
            self.awdbg_sampled_per_pos = None
        else:
            self.awdbg_sampled_per_pos = [0] * self.num_speculative_steps
        # ---- end GB10 C1 per-call counter ----
""",
        ),
        # ---- SP3b: C1 position-0 increment at the prefill rejoin (round 5)
        # ---- the first-pass sample has EXECUTED for every batch row by the
        # time the eager/captured-graph prefill branches rejoin; counting
        # there records completed work once per real call (the round-4
        # entry-side count pre-dated the prefill branches, and a count
        # inside the sampling step itself would miss FULL-graph replay).
        # num_reqs excludes graph padding. Placed BEFORE the
        # num_speculative_steps == 1 early return so a K=1 launch still
        # counts its position 0. ----
        (
            """        self.on_prefill_end(num_reqs)

        if self.num_speculative_steps == 1:
""",
            """        self.on_prefill_end(num_reqs)
        # ---- GB10 C1: position-0 samples counted at the prefill rejoin (P2
        # stage-1b round 5) ---- _awdbg_c1_pos0
        # The first-pass sample has executed for every batch row by HERE --
        # eagerly in _prefill or inside the captured prefill graph, either
        # way exactly one sample per row at draft position 0. Counting at
        # the rejoin records completed work once per real call, after the
        # eager/replay branches rejoin (an entry-side count would pre-date
        # the prefill branches; a count inside the sampling step itself
        # would miss FULL-graph replay). num_reqs excludes graph padding.
        if self.awdbg_sampled_per_pos is not None:
            self.awdbg_sampled_per_pos[0] += num_reqs

        if self.num_speculative_steps == 1:
""",
        ),
        # ---- SP4a: annotate the decode phase entry ----
        (
            """        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining num_speculative_steps - 1 draft tokens.
""",
            """        self.on_multi_step_decode_begin(num_reqs)
        # ---- GB10 adaptive width (P2 stage-1b): the decode loop bound is
        # self._aw_active_steps == k_eff (set at propose entry above).
        # Generate the remaining k_eff - 1 draft tokens.
""",
        ),
        # ---- SP4b: restore the launch bound + return the k_eff slice ----
        (
            """        self.on_multi_step_decode_end(num_reqs)

        return self.draft_tokens[:num_reqs]
""",
            """        self.on_multi_step_decode_end(num_reqs)
        # ---- GB10 adaptive width: restore the launch loop bound (P2
        # stage-1b) so capture/dummy paths always see the full width.
        self._aw_active_steps = self.num_speculative_steps

        # ---- GB10 adaptive width: the returned proposal slice honors k_eff
        # (P2 stage-1b) ---- a [num_reqs, k_eff] VIEW of the K_max-strided
        # buffer (row stride preserved, D-009 c); columns >= k_eff keep
        # stale content and are never read (masking contract -- the verify
        # window length equals the carried proposal width or a trim of it).
        return self.draft_tokens[:num_reqs, :k_eff]
""",
        ),
        # ---- SP5: the non-fused decode loop honors k_eff ----
        (
            """        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            # Rebuild every step when positions advance, or just once
""",
            """        attn_metadata = None
        slot_mappings_by_layer = None
        # ---- GB10 adaptive width: per-call loop bound (P2 stage-1b) ----
        for step in range(1, self._aw_active_steps):
            # Rebuild every step when positions advance, or just once
""",
        ),
        # ---- SP6: the fused decode loop honors k_eff ----
        (
            """        for step in range(1, self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(
""",
            """        # ---- GB10 adaptive width: per-call loop bound (P2 stage-1b) --
        # ---- eager fused loop; under FULL-graph replay the bound is baked
        # at capture (launch width) -- not this deployment's path.
        for step in range(1, self._aw_active_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(
""",
        ),
        # ---- SP7: C1 increment at the non-fused loop's eager sample ----
        (
            """            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )
""",
            """            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                # FULL-graph decode replay bakes its samples into the
                # captured graph -- not counted here (the same physical
                # limit class as the baked loop bound; this deployment runs
                # eager draft decode: PIECEWISE target graphs, fused draft
                # decode disabled for the QSA backend).
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )
                # ---- GB10 C1: one executed sample per batch row at draft
                # position `step` (P2 stage-1b).
                if self.awdbg_sampled_per_pos is not None:
                    self.awdbg_sampled_per_pos[step] += num_reqs
""",
        ),
        # ---- SP8: C1 increment at the fused loop's eager sample ----
        (
            """            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
""",
            """            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            # ---- GB10 C1: one executed sample per batch row at draft
            # position `step` (P2 stage-1b).
            if self.awdbg_sampled_per_pos is not None:
                self.awdbg_sampled_per_pos[step] += num_reqs
""",
        ),
    ],
    must_not_contain=[
        "_aw_active_steps",
        "num_spec_tokens_to_schedule: int | None",
        "_awdbg_c1_counter",
        "_awdbg_c1_pos0",
    ],
)

# ===========================================================================
# 4. spec_decode/metrics.py -- the C1 SpecDecodingStats field + the new
#    Prometheus counter (label family and position range copied from the
#    accepted-per-pos counter)
# ===========================================================================
patch(
    METRICS_PATH,
    edits=[
        # ---- MC1a: the SpecDecodingStats field ----
        (
            """    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)
""",
            """    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)
    # ---- GB10 C1 (P2 stage-1b): per-position executed drafter samples,
    # folded from the worker's DraftTokenIds.sampled_per_pos one engine
    # step after they executed (the post-step channel; the fold site is at
    # the end of Scheduler.update_from_output). Same sizing rule as the
    # accepted/draft per-pos vectors: launch-K long, zeros initialized.
    num_sampled_draft_tokens_per_pos: list[int] = field(default_factory=list)
""",
        ),
        # ---- MC1b: new() initializes it launch-K long ----
        (
            """        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_draft_tokens_per_pos=[0] * num_spec_tokens,
        )
""",
            """        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_draft_tokens_per_pos=[0] * num_spec_tokens,
            num_sampled_draft_tokens_per_pos=[0] * num_spec_tokens,
        )
""",
        ),
        # ---- MC2a: the counter dict slot (empty for diffusion boots) ----
        (
            """        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        if not is_diffusion:
""",
            """        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        # ---- GB10 C1 (P2 stage-1b): empty for diffusion boots (the
        # per-position sample vector does not apply there, like the
        # accepted-per-pos vector) -- observe() no-ops on the empty dict.
        self.counter_spec_decode_num_sampled_draft_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {}
        if not is_diffusion:
""",
        ),
        # ---- MC2b: create the counter next to the accepted-per-pos one ----
        (
            """            self.counter_spec_decode_num_accepted_tokens_per_pos = {
                idx: [
                    base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)
                ]
                for idx, lv in per_engine_labelvalues.items()
            }
""",
            """            self.counter_spec_decode_num_accepted_tokens_per_pos = {
                idx: [
                    base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)
                ]
                for idx, lv in per_engine_labelvalues.items()
            }
            # ---- GB10 C1 (P2 stage-1b): the executed-sample counter, same
            # label family (engine/model_name + position) and the same
            # launch-K position range, pre-created at init.
            awdbg_sample_counter = self._counter_cls(
                name="vllm:spec_decode_num_sampled_draft_tokens_per_pos",
                documentation="Actually-executed drafter samples per draft position.",
                labelnames=pos_labelnames,
            )
            self.counter_spec_decode_num_sampled_draft_tokens_per_pos = {
                idx: [
                    awdbg_sample_counter.labels(*lv, str(pos))
                    for pos in range(num_spec_tokens)
                ]
                for idx, lv in per_engine_labelvalues.items()
            }
""",
        ),
        # ---- MC3: observe() increments it per position ----
        (
            """        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos.get(engine_idx, [])
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
""",
            """        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos.get(engine_idx, [])
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
        # ---- GB10 C1 (P2 stage-1b): the executed-sample counter, next to
        # the accepted-per-pos counter. An empty per-position vector (a
        # SpecDecodingStats built without new()) increments nothing --
        # never an IndexError.
        awdbg_sampled_vec = spec_decoding_stats.num_sampled_draft_tokens_per_pos
        for pos, counter in enumerate(
            self.counter_spec_decode_num_sampled_draft_tokens_per_pos.get(
                engine_idx, []
            )
        ):
            if pos < len(awdbg_sampled_vec):
                counter.inc(awdbg_sampled_vec[pos])
""",
        ),
        # ---- MC4: the C1 fold can now produce zero-draft stats objects (a
        # chunked-prefill step's samples fold into the next step's stats,
        # which may have no verifies -- the proposals of prefill chunks are
        # discarded scheduler-side). log() aggregated over such a window
        # would divide by num_drafts == 0 (inf + RuntimeWarning spam in the
        # acceptance-length line); report nan instead, like the already
        # guarded draft_acceptance_rate two lines below. ----
        (
            """        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)
""",
            """        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        # ---- GB10 C1 (P2 stage-1b): a logging window can now aggregate
        # zero-draft stats objects (the C1 fold records executed samples on
        # verify-less steps, e.g. chunked prefill); nan, not inf.
        acceptance_rates = (
            np.sum(pos_matrix, axis=0) / num_drafts
            if num_drafts > 0
            else np.full(pos_matrix.shape[1], np.nan)
        )
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)
""",
        ),
        (
            """        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)
""",
            """        # Conventionally, mean acceptance length includes the bonus token
        # ---- GB10 C1 (P2 stage-1b): zero-draft windows report nan (see
        # the acceptance_rates guard above).
        mean_acceptance_length = (
            1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else float("nan")
        )
""",
        ),
    ],
    must_contain=["vllm:spec_decode_num_accepted_tokens_per_pos"],
    must_not_contain=[
        "num_sampled_draft_tokens_per_pos",
        "spec_decode_num_sampled_draft_tokens_per_pos",
    ],
)

# ===========================================================================
# 5. outputs.py -- DraftTokenIds carries the C1 counts (worker -> scheduler)
#    AND ModelRunnerOutput gains the same-step carrier field (round 5): the
#    async scheduler never pulls the post-step DraftTokenIds channel, so the
#    scheduler-side fold stages from the ModelRunnerOutput instead -- the
#    carrier that works on BOTH scheduler modes.
# ===========================================================================
patch(
    OUTPUTS_PATH,
    edits=[
        (
            """@dataclass
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]
""",
            """@dataclass
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]
    # ---- GB10 C1 (P2 stage-1b) ---- the worker's per-position executed
    # drafter-sample counts for the step that produced this proposal
    # (length == launch_k; the SAME list that feeds the worker evidence
    # line). None when the step ran no real speculator propose (dummy /
    # profile runs, speculator-less boots) -- the scheduler stages it and
    # folds it into SpecDecodingStats one step later (the post-step
    # channel; see the fold site at the end of update_from_output).
    sampled_per_pos: list[int] | None = None
""",
        ),
        # ---- OC1 (round 5): the ModelRunnerOutput C1 carrier ---- the V2
        # runner sets it right after propose (gpu_model_runner.py, the
        # _awdbg_c1_carrier site); Scheduler.update_from_output stages it
        # at the end of the step (AFTER the fold drained the previous
        # step's list -- the one-step lag is preserved). This carrier is
        # what keeps the C1 counter live under the ASYNC scheduler, which
        # never pulls the DraftTokenIds channel.
        (
            """    # ``None`` when ``return_sampling_mask`` is off.
    sampling_masks: SamplingMaskLists | None = None
""",
            """    # ``None`` when ``return_sampling_mask`` is off.
    sampling_masks: SamplingMaskLists | None = None

    # ---- GB10 C1 (P2 stage-1b round 5) ---- the worker's per-position
    # executed drafter-sample counts for THIS step (the speculator's own
    # counter; same list as DraftTokenIds.sampled_per_pos). None when the
    # step ran no real speculator propose (dummy/profile runs, warmup
    # excluded by the emitter gate, speculator-less boots). Staged by
    # Scheduler.update_from_output after that step's fold, so the counts
    # land in the NEXT step's SpecDecodingStats (the documented one-step
    # lag) on BOTH the sync and the async scheduler.
    sampled_per_pos: list[int] | None = None
""",
        ),
    ],
    must_contain=["class DraftTokenIds:"],
    must_not_contain=["sampled_per_pos"],
)

# ===========================================================================
# 6. spec_decode/utils.py -- DraftTokensHandler stages + ships the C1 counts
#    on BOTH paths (structured-output copy path AND the placeholder path)
# ===========================================================================
patch(
    SDUTILS_PATH,
    edits=[
        # ---- UC1: handler state + the set_draft_tokens signature ----
        (
            """        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
""",
            """        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        # ---- GB10 C1 (P2 stage-1b): the worker's per-position executed
        # drafter-sample counts for the CURRENT step (staged by
        # set_draft_tokens, shipped by get_draft_tokens on BOTH paths --
        # including the no-structured-output placeholder path).
        self.sampled_per_pos: list[int] | None = None

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        sampled_per_pos: list[int] | None = None,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        # ---- GB10 C1 (P2 stage-1b) ---- None on a dummy/profile propose
        # or a speculator-less boot; a real propose always stages a fresh
        # launch-K list, so a stale list is never re-shipped.
        self.sampled_per_pos = sampled_per_pos
""",
        ),
        # ---- UC2: get_draft_tokens ships the counts on both paths ----
        (
            """        return DraftTokenIds(self.req_ids, draft_token_ids)
""",
            """        # ---- GB10 C1 (P2 stage-1b) ---- the executed-sample counts
        # ride BOTH paths (the structured-output copy path AND this
        # placeholder path, which every non-structured-output boot takes):
        # the gate's C1 counter checks consume them per step.
        return DraftTokenIds(self.req_ids, draft_token_ids, self.sampled_per_pos)
""",
        ),
    ],
    must_contain=["class DraftTokensHandler:", "def get_draft_tokens("],
    must_not_contain=["sampled_per_pos"],
)

print("patch_width_runner: all six files patched OK")
