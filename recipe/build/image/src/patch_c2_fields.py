#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P2 stage-1 closure: C2 emitter FIELD completion (D-014 condition 2).

Applied ON TOP of the iter7d image state (stage-1a/1c + the stage-1b
six-file chain + the moe-align stable-sort shim are already in). The
round-8 astra (b) design completes the emitter record so the gate can join
PROPOSALS (worker side) with VERIFY EVENTS and COMMITS (scheduler side)
per request:

  worker line gains (per scheduled propose):
    branch           the executed draft-sampling path ("gumbel" = the
                     proposal-logit-cache probabilistic path -- the branch
                     sample_draft takes whenever draft_logits is not None;
                     "greedy" = the argmax path), recorded at propose
                     entry in autoregressive/speculator.py,
    nt / pf          execution-shape fingerprint (scheduled token count,
                     chunked-prefill presence),
    reqs[].temp / seed / seed_set
                     the drafter's EFFECTIVE per-row sampling temperature /
                     RNG seed / explicit-seed flag, read from the sampler's
                     CPU mirrors (SamplingStates.temperature.np / .seeds.np
                     / .seeds_set) at the req_state row the proposal
                     buffers are addressed by -- NO GPU sync,
    reqs[].props     the drafter's actual proposal token ids per row (the
                     k_eff stored columns; ONE evidence-gated GPU->host
                     transfer -- the handler's placeholder path never
                     carries them on gate boots).

  sched line gains (per scheduled step, in Scheduler.update_from_output):
    reqs[].rej_idx   the first rejected position of the verify (= the
                     accepted count; the bonus position on a full
                     acceptance); null without a verify window,
    reqs[].bonus     the target's own sampled token at rej_idx (the
                     replacement on a rejection, the bonus token on a full
                     acceptance),
    reqs[].emit      the tokens actually committed this step (exactly what
                     entered the request's output list, stop-trims
                     reflected),
    reqs[].stopped   whether THIS commit stopped the request (bool(stopped)
                     as of the stop-check chain -- the _update_request_
                     with_output stop-trim or the pooling/encoder arms;
                     the stop-trim harness's termination proof),
    reqs[].phash     committed-prefix hash: sha256 over the request's whole
                     committed token prefix (prompt + output, int64-le)
                     AFTER the append -- maintained incrementally per
                     request in awdbg1b_evidence.commit_hash_update,
                     re-based on a length regression,
    reqs[].ol / nct  semantic cache-state fingerprint pair (committed
                     output length; num_computed_tokens post-rollback),
    reqs[].temp / seed / seed_set
                     the sampling params AS SCHEDULED (the scheduler-side
                     mirror the harness join cross-checks).

All fields are ADDITIVE: an old-shape line stays parse-compatible; the
harness's new C2-field checks report MISSING_INSTRUMENT on absent fields
and MISSING_MALFORMED_INSTRUMENT on wrongly-typed values (never a crash).
The per-step cost stays evidence-gated (VLLM_SPECDBG_EVIDENCE=1) plus two
bounded host transfers per scheduled step (idx rows already synced before;
props is the one new transfer).

Anchored, assert-guarded edits (every anchor must match exactly once; every
edited file is ast.parse-checked afterwards):

  scheduler.py (vllm/v1/core/sched/scheduler.py)
    - SC1 plumbing gains the per-step verify/commit/sampling stashes and
      the persistent per-request prefix-hash book,
    - the SC2 verify-count site also stashes the verify record (rej_idx)
      and the target's token at that position (the bonus/replacement),
    - a post-append capture site -- placed AFTER the whole stop-check
      if/elif chain so the chain's elif arms keep chaining to the original
      if -- records emit/stopped/phash/ol/nct + the sampling mirror for
      EVERY request that committed tokens this step (a prefill row's first
      token commits too -- not gated on the spec block),
    - the C2 SCHED emitter call passes the three new maps and prunes the
      hash book down to the live requests.

  autoregressive/speculator.py
    - propose() records the executed draft-sampling branch (the same
      condition base sample_draft tests per call; the proposal-logit
      buffer never changes mid-call).

  gpu/model_runner.py (V2)
    - the C2 WORKER emitter call gains branch/nt/pf and the per-row
      temp/seed/seed_set/props fields.

Usage:
    python3 patch_c2_fields.py <scheduler.py> <gpu_model_runner.py> \
        <autoregressive_speculator.py>
"""
import ast
import sys

if len(sys.argv) != 4:
    print(
        "usage: patch_c2_fields.py <scheduler.py> <gpu_model_runner.py> "
        "<autoregressive_speculator.py>",
        file=sys.stderr,
    )
    sys.exit(64)

SCHED_PATH = sys.argv[1]
RUNNER_PATH = sys.argv[2]
SPEC_PATH = sys.argv[3]


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
    ast.parse(s)  # BEFORE the write: a parse failure must not leave the
    # target file mutated on disk (the reviewer's write-then-parse finding)
    open(path, "w").write(s)
    print(f"patch_c2_fields: {path} OK ({len(edits)} edits, ast.parse pass)")


# ---------------------------------------------------------------------------
# The CF3 chain-preservation assertion: the commit capture must sit AFTER
# the complete stop-check if/elif chain, never between the if block and its
# elif arms (that would chain the arms to the inserted evidence if and make
# the pooling/encoder stop checks dead whenever evidence is enabled).
# ---------------------------------------------------------------------------
def _stop_check_chain_heads(tree):
    """The `if new_token_ids:` stop-check chain heads in one parsed tree.

    Narrow discriminator: a bare ``if new_token_ids:`` (a plain Name test)
    whose FIRST body statement assigns the ``_update_request_with_output``
    result. The scheduler has exactly one such chain; the assertions below
    fail loudly when that stops holding. Takes the TREE (not source) so the
    returned nodes are identity-comparable within that one parse -- the
    sibling lookup below relies on that.
    """
    heads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name)
                and node.test.id == "new_token_ids"):
            continue
        first = node.body[0] if node.body else None
        if not (isinstance(first, ast.Assign)
                and isinstance(first.value, ast.Call)):
            continue
        fn = first.value.func
        if isinstance(fn, ast.Attribute) \
                and fn.attr == "_update_request_with_output":
            heads.append(node)
    return heads


def _statement_after(tree, node):
    """The sibling statement following `node` in its body (or None)."""
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(parent, field, None)
            if isinstance(seq, list):
                for i, child in enumerate(seq):
                    if child is node:
                        return seq[i + 1] if i + 1 < len(seq) else None
    return None


def _is_capture_if(node):
    """True for the CF3 commit-capture If: ``if awdbg_ev_enabled():``."""
    return (isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "awdbg_ev_enabled"
            and not node.test.args and not node.test.keywords)


def _capture_records_stopped(node):
    """True when the capture If's awdbg_commit record carries "stopped"."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name)
            and t.value.id == "awdbg_commit"
            for t in sub.targets
        ):
            continue
        if isinstance(sub.value, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "stopped"
            for k in sub.value.keys
        ):
            return True
    return False


def assert_c2_chain_preserved(before_src, after_src):
    """Fail loudly unless the patched scheduler kept the stop-check chain.

    Asserted on the patched file against the pre-patch source:
      1. the chain head still exists, exactly once;
      2. the whole chain subtree is ast.dump-identical to the pre-patch one
         (same head body, same elif sequence -- the CF3 replacement
         reproduces the chain bytes exactly; ANY change inside the chain
         fails here and must be re-reviewed consciously);
      3. the commit capture (test = awdbg_ev_enabled()) is the sibling
         statement immediately AFTER the complete chain, never inside it;
      4. the capture's awdbg_commit record carries the "stopped" field
         (bool(stopped) -- the stop-trim harness's termination proof);
      5. the pre-patch source had no capture after the chain (the check
         really discriminates pre from post).
    """
    before_tree = ast.parse(before_src)
    after_tree = ast.parse(after_src)
    before_heads = _stop_check_chain_heads(before_tree)
    after_heads = _stop_check_chain_heads(after_tree)
    assert len(before_heads) == 1, (
        "c2 chain check: expected exactly one stop-check chain in the "
        f"pre-patch scheduler, found {len(before_heads)}")
    assert len(after_heads) == 1, (
        "c2 chain check: expected exactly one stop-check chain in the "
        f"patched scheduler, found {len(after_heads)}")
    assert ast.dump(before_heads[0]) == ast.dump(after_heads[0]), (
        "c2 chain check: the stop-check if/elif chain changed shape -- the "
        "CF3 capture must not sit between the if block and its elif arms")
    after_next = _statement_after(after_tree, after_heads[0])
    assert _is_capture_if(after_next), (
        "c2 chain check: the commit capture is not the sibling statement "
        "right after the complete stop-check chain")
    assert _capture_records_stopped(after_next), (
        "c2 chain check: the awdbg_commit record has no stopped field")
    before_next = _statement_after(before_tree, before_heads[0])
    assert not _is_capture_if(before_next), (
        "c2 chain check: the pre-patch scheduler already had a capture "
        "after the chain")


# ===========================================================================
# 1. scheduler.py -- verify/commit stashes, the post-append commit capture,
#    and the extended C2 SCHED emitter call
# ===========================================================================
sched_before = open(SCHED_PATH).read()
patch(
    SCHED_PATH,
    edits=[
        # ---- CF1: the SC1 plumbing gains the three per-step stashes and
        # the persistent prefix-hash book ----
        (
            """        # Per-request (accepted, rejected) draft counts for this step's
        # sched evidence line; requests without a spec verify stay (0, 0).
        awdbg_acc: dict[str, tuple[int, int]] = {}
        # ---- end GB10 AWDBG1B sched evidence plumbing ----
""",
            """        # Per-request (accepted, rejected) draft counts for this step's
        # sched evidence line; requests without a spec verify stay (0, 0).
        awdbg_acc: dict[str, tuple[int, int]] = {}
        # ---- GB10 AWDBG1B (round 8, C2 field completion): the verify and
        # commit stashes ---- _awdbg_c2_field_stashes
        # awdbg_verify: the verify-side record per request WITH a verify
        # window (rej_idx = first rejected position = the accepted count;
        # bonus = the target's own sampled token at that position, captured
        # at the stash site below from the verify output). awdbg_commit:
        # what each request COMMITTED this step (emit / phash / ol / nct),
        # captured after the append. awdbg_sampling: the sampling params AS
        # SCHEDULED (the scheduler-side mirror the harness join
        # cross-checks against the worker's). awdbg_book: the PERSISTENT
        # per-request committed-prefix hash state (one sha256 hasher + the
        # hashed length per live request; fed incrementally, re-based on a
        # length regression; pruned at the emitter site down to the live
        # requests so the book never outlives the request set).
        awdbg_verify: dict[str, dict] = {}
        awdbg_bonus: dict[str, int | None] = {}
        awdbg_commit: dict[str, dict] = {}
        awdbg_sampling: dict[str, dict] = {}
        awdbg_book = getattr(self, "_awdbg_commit_book", None)
        if awdbg_book is None:
            awdbg_book = self._awdbg_commit_book = {}
        # ---- end GB10 AWDBG1B sched evidence plumbing ----
""",
        ),
        # ---- CF2: the SC2 verify site also stashes rej_idx + the target's
        # token at that position (generated_token_ids[num_accepted] is the
        # verify output's token AT the rejection/bonus position: the
        # replacement on a rejection, the bonus on a full acceptance;
        # single-sample steps -- num_sampled == 1 -- make it the last token
        # too; when the output list runs longer than num_accepted the token
        # AT that index is taken (exact for the single-sample MTP path this
        # deployment runs; a multi-sample request would need its own rule)) ----
        (
            """                if awdbg_ev_enabled():
                    awdbg_acc[req_id] = (int(num_accepted), int(num_rejected))
                # ---- end GB10 AWDBG1B verify counts ----
""",
            """                if awdbg_ev_enabled():
                    awdbg_acc[req_id] = (int(num_accepted), int(num_rejected))
                    # ---- GB10 AWDBG1B (round 8): the verify-side record
                    # ---- rej_idx = num_accepted: the first position whose
                    # draft token the target did NOT accept (== the window
                    # length when every position accepted -- the bonus
                    # position). bonus = the target's own sampled token at
                    # that position (the verify output's token AT index
                    # num_accepted; the replacement on a rejection, the
                    # bonus token on a full acceptance).
                    awdbg_verify[req_id] = {"rej_idx": int(num_accepted)}
                    awdbg_bonus[req_id] = (
                        int(generated_token_ids[num_accepted])
                        if len(generated_token_ids) > num_accepted
                        else None
                    )
                # ---- end GB10 AWDBG1B verify counts ----
""",
        ),
        # ---- CF3: the post-append commit capture (EVERY request that
        # committed tokens this step -- a prefill row's first token commits
        # too, so this is NOT gated on the spec-verify block above). The
        # anchor spans the WHOLE stop-check if/elif chain (the
        # _update_request_with_output if + the pooling and encoder-only
        # elif arms) and the capture is appended AFTER it as an independent
        # sibling statement: anchoring after the if-block alone made the
        # elif arms chain to the inserted evidence if, so with evidence
        # enabled the pooling/encoder stop checks were dead. The chain text
        # is reproduced byte-for-byte and assert_c2_chain_preserved (below)
        # re-proves the structure on the patched file. The capture point
        # must also stay AFTER the chain, never before it: the tokens it
        # reads are appended only inside _update_request_with_output (the
        # append_output_token_ids loop), and "stopped" is final only after
        # the elif arms have had their turn ----
        (
            """            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            elif (
                self.is_encoder_only
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
""",
            """            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            elif (
                self.is_encoder_only
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            # ---- GB10 AWDBG1B: per-request commit capture (P2 stage-1b
            # round 8, C2 field completion) ---- _awdbg_commit_capture
            # AFTER the whole stop-check chain (NOT between the if block and
            # its elif arms -- that would chain the arms to THIS if and kill
            # the pooling/encoder stop checks whenever evidence is on; see
            # assert_c2_chain_preserved below): the tokens that actually
            # entered the request's committed output exist only once
            # _update_request_with_output has appended them (the
            # append_output_token_ids loop inside that call; exactly
            # request._output_token_ids' new tail, so a stop-trim is
            # reflected), "stopped" is the chain's verdict (the
            # _update_request_with_output stop-trim or the pooling/encoder
            # arms -- the grammar-error stop further below is NOT part of
            # this field), recorded as the stop-trim harness's termination
            # proof for a shortened emit tail, the committed-prefix hash
            # advanced over the request's WHOLE committed token prefix
            # (prompt + output = request._all_token_ids; int64-le encoding,
            # fed incrementally, re-based from scratch if the committed
            # length ever regresses), and the semantic cache-state pair
            # (ol = committed output length, nct = num_computed_tokens
            # post-rollback). The sampling mirror records the params AS
            # SCHEDULED (seed 0 + seed_set False when the request set
            # none -- the engine fills an unset seed randomly worker-side,
            # so the harness compares seeds only when both sides set one).
            if awdbg_ev_enabled():
                awdbg_commit[req_id] = {
                    "emit": list(
                        request._output_token_ids[num_output_tokens_before:]
                    ),
                    "stopped": bool(stopped),
                    # the finishing status name (FINISHED_LENGTH_CAPPED /
                    # FINISHED_STOPPED / FINISHED_ABORTED) when this commit
                    # stopped the request; None otherwise -- the harness's
                    # length-cap label reads the substring "length"
                    "stop_reason": (
                        getattr(request.status, "name", None)
                        if stopped else None
                    ),
                    "phash": awdbg_commit_hash(
                        awdbg_book, req_id, request._all_token_ids
                    )[0],
                    "ol": len(request._output_token_ids),
                    "nct": int(request.num_computed_tokens),
                }
                awdbg_sp = request.sampling_params
                awdbg_sampling[req_id] = {
                    "temp": (float(awdbg_sp.temperature)
                             if awdbg_sp is not None else None),
                    "seed": (int(awdbg_sp.seed)
                             if awdbg_sp is not None
                             and awdbg_sp.seed is not None else 0),
                    "seed_set": bool(
                        awdbg_sp is not None and awdbg_sp.seed is not None),
                }
            # ---- end GB10 AWDBG1B commit capture ----
""",
        ),
        # ---- CF4: the SC1 lazy import gains the commit-hash helper ----
        (
            """        from awdbg1b_evidence import (  # noqa: E402 - evidence-gated helper
            evidence_enabled as awdbg_ev_enabled,
        )
""",
            """        from awdbg1b_evidence import (  # noqa: E402 - evidence-gated helper
            commit_hash_update as awdbg_commit_hash,
            evidence_enabled as awdbg_ev_enabled,
        )
""",
        ),
        # ---- CF5: the C2 SCHED emitter call passes the three new maps and
        # prunes the hash book down to the live requests ----
        (
            """            _awdbg_emit(
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
""",
            """            # ---- GB10 AWDBG1B (round 8): merge the bonus tokens into
            # the verify records and prune the prefix-hash book down to the
            # still-tracked requests (finished/aborted requests' hashers
            # must not outlive them) ---- _awdbg_c2_emit_prep
            for awdbg_rid, awdbg_v in awdbg_verify.items():
                awdbg_v["bonus"] = awdbg_bonus.get(awdbg_rid)
            awdbg_commit_prune(
                awdbg_book, self.requests.keys()
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
                    verify=awdbg_verify,
                    commit=awdbg_commit,
                    sampling=awdbg_sampling,
                ),
            )
""",
        ),
        # ---- CF6: the emitter's lazy import gains the prune helper ----
        (
            """            from awdbg1b_evidence import (  # noqa: E402 - evidence-gated
                build_sched_record as _awdbg_bsr,
                emit_line as _awdbg_emit,
            )
""",
            """            from awdbg1b_evidence import (  # noqa: E402 - evidence-gated
                build_sched_record as _awdbg_bsr,
                commit_hash_prune as awdbg_commit_prune,
                emit_line as _awdbg_emit,
            )
""",
        ),
    ],
    must_contain=[
        "_awdbg_sched_emitter",
        "_awdbg_c1_fold",
        "_adaptive_width_armed_branch",
    ],
    must_not_contain=[
        "_awdbg_c2_field_stashes",
        "_awdbg_commit_capture",
        "_awdbg_c2_emit_prep",
        "awdbg_draft_branch",
    ],
)
# the chain-preservation proof runs on the REAL patched scheduler bytes
assert_c2_chain_preserved(sched_before, open(SCHED_PATH).read())

# ===========================================================================
# 2. autoregressive/speculator.py -- the executed draft-sampling branch
# ===========================================================================
patch(
    SPEC_PATH,
    edits=[
        (
            """        if dummy_run or is_profile:
            self.awdbg_sampled_per_pos = None
        else:
            self.awdbg_sampled_per_pos = [0] * self.num_speculative_steps
        # ---- end GB10 C1 per-call counter ----
""",
            """        if dummy_run or is_profile:
            self.awdbg_sampled_per_pos = None
        else:
            self.awdbg_sampled_per_pos = [0] * self.num_speculative_steps
        # ---- end GB10 C1 per-call counter ----
        # ---- GB10 AWDBG1B (round 8): the executed draft-sampling branch
        # ---- _awdbg_draft_branch
        # The SAME condition the base sample_draft tests on every call:
        # the proposal-logit cache is allocated iff draft_sample_method ==
        # "probabilistic", and sample_draft takes the gumbel path whenever
        # draft_logits is not None (else the argmax path). Recorded at
        # propose entry; the proposal-logit buffer never changes mid-call,
        # so the value describes every sample this propose executed. The
        # C2 worker line reports it per scheduled propose.
        self.awdbg_draft_branch = (
            "gumbel" if self.draft_logits is not None else "greedy"
        )
""",
        ),
    ],
    must_contain=["_awdbg_c1_counter"],
    must_not_contain=["_awdbg_draft_branch"],
)

# ===========================================================================
# 3. gpu/model_runner.py (V2) -- the C2 WORKER emitter call gains the
#    proposal-side fields
# ===========================================================================
patch(
    RUNNER_PATH,
    edits=[
        (
            """                if _awdbg_ev():
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
""",
            """                if _awdbg_ev():
                    # ---- GB10 AWDBG1B (round 8): the proposal-side fields
                    # ---- rows: the req_state rows (the same addresses the
                    # proposal buffers use). temps/seeds/seed_set: read from
                    # the sampler's CPU mirrors (UvaBackedTensor.np /
                    # .seeds_set numpy arrays) at those rows -- NO GPU sync;
                    # these are the buffers THIS propose consumed
                    # (_copy_request_inputs copies them into the drafter's
                    # own buffers before sampling). props: the stored k_eff
                    # proposal columns -- ONE evidence-gated GPU->host
                    # transfer (the DraftTokensHandler placeholder path
                    # never carries real ids on gate boots, so this is the
                    # only host copy of the proposal). branch: the executed
                    # draft-sampling path recorded by the speculator.
                    _awdbg_rows = input_batch.idx_mapping.tolist()
                    _awdbg_states = self.sampler.sampling_states
                    _awdbg_emit(
                        logger,
                        _awdbg_bwr(
                            k_eff=int(draft_tokens.shape[1]),
                            launch_k=self.num_speculative_steps,
                            sampled_per_pos=self.speculator.awdbg_sampled_per_pos,
                            req_ids=input_batch.req_ids,
                            idx_rows=_awdbg_rows,
                            branch=getattr(
                                self.speculator, "awdbg_draft_branch", None
                            ),
                            num_tokens=int(input_batch.num_tokens),
                            has_prefill=bool(input_batch.has_prefill),
                            temps=[
                                float(_awdbg_states.temperature.np[r])
                                for r in _awdbg_rows
                            ],
                            seeds=[
                                int(_awdbg_states.seeds.np[r])
                                for r in _awdbg_rows
                            ],
                            seeds_set=[
                                bool(_awdbg_states.seeds_set[r])
                                for r in _awdbg_rows
                            ],
                            props=draft_tokens.tolist(),
                        ),
                    )
""",
        ),
    ],
    must_contain=["_awdbg_worker_emitter"],
    must_not_contain=["_awdbg_states", "awdbg_draft_branch"],
)

print("patch_c2_fields: all three files patched OK")
