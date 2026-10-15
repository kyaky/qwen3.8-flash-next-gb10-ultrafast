# SPDX-License-Identifier: Apache-2.0
"""AWDBG1B evidence emitters for the P2 stage-1b gate (shared helper).

Pure stdlib: importable by the patched vllm files (the image copies it next
to them in site-packages) AND by the CPU self-test
(src/test_width_runner_cpu.py) with no vllm/torch imports. The JSON
construction lives here so the gate harness's parse_evidence sees EXACTLY
the records the live emitters build -- the self-test proves the format
compatibility before any boot.

C2 contract (agents/glm-5.3-flash-integrator/scripts/gate-stage1b/README.md,
sections C2 + C2-alignment + C2-fields; record shapes mirror the harness
selftest's MockWorld emission code in driver.py):

  worker: AWDBG1B {"src":"worker","seq":N,"t":<unix_us>,"k_eff":K,
                   "sampled_per_pos":[<int x launch_k>],
                   "branch":"gumbel"|"greedy","nt":<int>,"pf":<bool>,
                   "reqs":[{"id","valid_len","row","steps",
                            "temp","seed","seed_set","props"}, ...]}
  sched:  AWDBG1B {"src":"sched","seq":N,"t":<unix_us>,"launch_k":L,
                   "requested_k":K,"bs":B,"mode":"<controller mode>",
                   "reqs":[{"id","win","acc","rej",
                            "rej_idx","bonus","emit","stopped",
                            "phash","ol","nct",
                            "temp","seed","seed_set"}, ...]}

Round 8 (P2 stage-1 closure, D-014 condition 2 -- the C2 emitter field
completion per the astra (b) design; ALL new fields are ADDITIVE: an
old-shape line stays parse-compatible and the harness's new checks report
MISSING_INSTRUMENT on the absent fields, never a crash):

  worker (per scheduled propose -- the PROPOSAL side):
    branch      the draft-sampling path the drafter executed for this call:
                "gumbel" (the proposal-logit-cache probabilistic path --
                the branch sample_draft takes whenever draft_logits is not
                None) or "greedy" (the argmax path). Recorded at propose
                entry from the SAME condition the sampler tests.
    nt / pf     execution-shape fingerprint of the step: scheduled token
                count and chunked-prefill presence (input_batch.num_tokens
                / input_batch.has_prefill).
    reqs[].temp / seed / seed_set
                the drafter's EFFECTIVE per-row sampling temperature, RNG
                seed and explicit-seed flag, read from the sampler's CPU
                mirrors (SamplingStates.temperature.np / .seeds.np /
                .seeds_set) at the req_state row the proposal buffers are
                addressed by -- the same buffers this propose consumed.
                NO GPU sync (numpy mirrors; a seed the user did not set is
                the engine's random filler, flagged seed_set=false).
    reqs[].props
                the drafter's actual proposal token ids for that row, in
                draft-step order -- the k_eff columns the runner stores
                (draft_tokens[row]). WORKER TRUTH for the proposal side.

  sched (per scheduled step -- the VERIFY/COMMIT side; one step after the
  proposal it verifies, the documented lag):
    reqs[].rej_idx
                the first rejected position of this step's verify: the
                number of accepted draft tokens (= num_accepted); when
                every draft position accepted it equals the window length
                (the bonus position). null when the request had no verify
                window this step.
    reqs[].bonus
                the target's own sampled token at rej_idx -- the
                replacement token on a rejection, the bonus token on a
                full acceptance (the LAST token of the verify output).
                null when no verify window.
    reqs[].emit the token ids actually committed to the request this step
                (accepted drafts + the final sample, exactly what entered
                the request's output token list, stop-trims reflected).
                [] when nothing was committed (prefill chunk steps).
    reqs[].stopped
                whether THIS commit stopped the request: bool(stopped) as
                of the stop-check chain -- the _update_request_with_output
                stop-trim or the pooling/encoder stop arms (the later
                grammar-error stop is not part of this field). The
                stop-trim harness reads it as the termination proof for a
                shortened emit tail. ADDITIVE: absent on old-shape lines.
    reqs[].phash
                committed-prefix hash: sha256 over the request's whole
                committed token prefix (prompt + output, int64-le byte
                encoding) AFTER this step's append -- maintained
                incrementally per request (feed the delta), re-based from
                scratch if the committed length ever regresses. The
                cross-boot matched-control comparison joins on it.
    reqs[].ol / nct
                semantic cache-state fingerprint pair after the step:
                committed output-token count and num_computed_tokens
                (post-rollback; the request's cache position).
    reqs[].temp / seed / seed_set
                the request's sampling params AS SCHEDULED (the
                scheduler-side mirror of what the worker's buffers hold);
                the harness join cross-checks worker-vs-sched.

  event/proposal joins: the harness pairs worker seq N-1's proposals with
  sched seq N's verify events per request (the one-step lag) and asserts,
  for greedy rows (temp == 0), the target-logit/argmax evidence:
  emit[:rej_idx] == props[:rej_idx] (the target's greedy verify accepted
  exactly the drafter's argmax prefix) and emit[rej_idx] == bonus with
  bonus != props[rej_idx] on a true rejection.

Both emitters keep their OWN monotonic step-sequence counter (per source,
per process), so the harness's join_by_seq pairs one worker line with one
sched line per engine step. Gated by VLLM_SPECDBG_EVIDENCE=1 (one env read,
cached per process; default 0 = zero overhead).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time

TAG = "AWDBG1B"
ENV_EVIDENCE = "VLLM_SPECDBG_EVIDENCE"

# Per-process, per-source step-sequence counters ("monotonic int per emitter
# source"): worker and sched each count engine steps independently, so equal
# seq values pair the two lines of one engine step in join_by_seq.
_SEQS: dict[str, int] = {}
_ENABLED: bool | None = None


def evidence_enabled() -> bool:
    """Env gate for both emitters; read once per process (default off)."""
    global _ENABLED
    if _ENABLED is None:
        raw = os.environ.get(ENV_EVIDENCE, "0")
        _ENABLED = raw.strip() not in ("", "0")
    return _ENABLED


def next_seq(source: str) -> int:
    """Next monotonic step-sequence value for one emitter source."""
    _SEQS[source] = _SEQS.get(source, 0) + 1
    return _SEQS[source]


def _now_us() -> int:
    return int(time.time() * 1_000_000)


# ---------------------------------------------------------------------------
# committed-prefix hash book (sched side; pure stdlib so the CPU self-test
# exercises the exact implementation the live scheduler runs)
# ---------------------------------------------------------------------------
_PHASH_ENCODING = struct.Struct("<q")  # int64 little-endian per token id


def commit_hash_update(book: dict, req_id: str, all_token_ids) -> tuple[str, int]:
    """Advance (or create) the request's committed-prefix hash.

    `book` maps req_id -> [sha256 hasher, hashed_len]; `all_token_ids` is the
    request's committed token prefix AFTER this step's append (prompt +
    output, the Request._all_token_ids list). The hasher is fed only the
    delta beyond hashed_len; a length REGRESSION (the committed prefix was
    truncated -- preemption mechanics) re-creates the hasher and re-feeds
    the whole prefix, so the digest always describes the CURRENT prefix.

    Returns (hexdigest, prefix_length).
    """
    n = len(all_token_ids)
    entry = book.get(req_id)
    if entry is None or n < entry[1]:
        h = hashlib.sha256()
        entry = [h, 0]
        book[req_id] = entry
    h, hashed = entry
    if n > hashed:
        delta = all_token_ids[hashed:]
        # first feed covers the whole prefix (prompt + output so far);
        # later feeds cover only the newly committed tokens
        h.update(b"".join(_PHASH_ENCODING.pack(int(t)) for t in delta))
        entry[1] = n
    return h.hexdigest(), n


def commit_hash_prune(book: dict, live_ids) -> int:
    """Drop book entries for requests no longer tracked (finished/aborted).

    Returns the number of entries dropped. Bounded state: the book only ever
    holds ACTIVE requests' hashers (one sha256 object + an int each).
    """
    live = set(live_ids)
    dead = [rid for rid in book if rid not in live]
    for rid in dead:
        del book[rid]
    return len(dead)


def build_worker_record(
    k_eff: int,
    launch_k: int,
    sampled_per_pos,  # sequence[int], length launch_k -- the speculator's
                      #   executed-sample counter (C1); never None here (a
                      #   real propose always stages a fresh launch-K list)
    req_ids,          # sequence[str] -- input_batch order
    idx_rows,         # sequence[int] -- req_state row per batch row
    source: str = "worker",
    branch: str | None = None,
    num_tokens: int | None = None,
    has_prefill: bool | None = None,
    temps=None,       # sequence[float] per row -- effective draft temperature
    seeds=None,       # sequence[int] per row -- effective RNG seed
    seeds_set=None,   # sequence[bool] per row -- seed explicitly set
    props=None,       # sequence[sequence[int]] per row -- proposal token ids
) -> dict:
    """Worker-side C2 record: one per non-dummy speculator.propose call.

    WORKER TRUTH (round 4): the ar drafter executes one sample per batch
    row at EVERY position 0..k_eff-1 (the first-pass sample_draft plus each
    _generate_draft call in the decode loop), so

    - sampled_per_pos is the speculator's OWN executed-sample counter for
      this step (the SAME list that feeds the C1 Prometheus counter via
      DraftTokenIds.sampled_per_pos) -- on this drafter that is
      [num_reqs]*k_eff followed by zeros. It is NOT derived from
      scheduler-visible windows: scheduler counters cannot distinguish
      "generated short" from "generated wide then trimmed", which is the
      discrimination D-009 demands (the trim discriminator itself survives
      through k_eff = draft_tokens.shape[1], genuine worker truth).
    - valid_len = k_eff for EVERY row: every batch row samples, including
      rows without a scheduled verify window (chunked prefill consumes its
      samples; its windows stay empty scheduler-side).
    - steps = range(valid_len) (contiguous draft-step columns, each
      < k_eff -- the proposal-logit cache alignment contract; NOTE this is
      the emitter's reporting convention, not an engine observation).
    - reqs[]: EVERY batch row, in input_batch order (NOT id-sorted -- the
      C2 order contract; a canonical emitter blinds P2.B4b/B4c);
      row = the req_state row the proposal buffers are addressed by
      (int(input_batch.idx_mapping[j])).

    Round-8 proposal-side fields (all optional -- an emitter that cannot
    supply them yet simply omits them, which the harness's C2-field checks
    report as MISSING_INSTRUMENT; an emitter that supplies a wrongly-typed
    value is reported as MISSING_MALFORMED_INSTRUMENT, never a crash):
      - branch: the executed draft-sampling path ("gumbel"/"greedy").
      - num_tokens/has_prefill: the step's execution-shape fingerprint.
      - temps/seeds/seeds_set: per-row effective sampling params from the
        sampler's CPU mirrors (no GPU sync).
      - props: per-row proposal token ids (the k_eff stored columns).
    """
    k_eff = int(k_eff)
    launch_k = int(launch_k)
    sampled = [int(v) for v in sampled_per_pos]
    assert len(sampled) == launch_k, (
        f"sampled_per_pos length {len(sampled)} != launch_k {launch_k}"
    )
    reqs: list[dict] = []
    for j, rid in enumerate(req_ids):
        row = int(idx_rows[j]) if j < len(idx_rows) else -1
        rq: dict = {
            "id": rid,
            "valid_len": k_eff,
            "row": row,
            "steps": list(range(k_eff)),
        }
        if temps is not None:
            rq["temp"] = float(temps[j])
        if seeds is not None:
            rq["seed"] = int(seeds[j])
        if seeds_set is not None:
            rq["seed_set"] = bool(seeds_set[j])
        if props is not None:
            rq["props"] = [int(t) for t in props[j]]
        reqs.append(rq)
    rec: dict = {
        "src": source,
        "seq": next_seq(source),
        "t": _now_us(),
        "k_eff": k_eff,
        "sampled_per_pos": sampled,
        "reqs": reqs,
    }
    if branch is not None:
        rec["branch"] = str(branch)
    if num_tokens is not None:
        rec["nt"] = int(num_tokens)
    if has_prefill is not None:
        rec["pf"] = bool(has_prefill)
    return rec


def build_sched_record(
    launch_k: int,
    requested_k: int,
    scheduled_req_ids,  # sequence[str] -- scheduler scheduled order
    spec_windows,       # mapping req_id -> verify-window list
    acc_rej,            # mapping req_id -> (accepted, rejected) counts
    mode: str,
    source: str = "sched",
    verify=None,        # mapping req_id -> {"rej_idx","bonus"} (verify side)
    commit=None,        # mapping req_id -> {"emit","stopped","phash","ol","nct"}
    sampling=None,      # mapping req_id -> {"temp","seed","seed_set"}
) -> dict:
    """Scheduler-side C2 record: one per update_from_output (non-empty step).

    - requested_k: the num_spec_tokens_to_schedule shipped in THIS step's
      SchedulerOutput (the post-armed-branch value).
    - win: len(scheduled_spec_decode_tokens[req_id]) -- the pending verify
      window (one-step lag; 0 for rows without a window, e.g. chunked
      prefill).
    - acc/rej: the per-request accepted/rejected counts already computed in
      update_from_output (0 when the request had no spec verify this step).
    - reqs order: the scheduler's scheduled order (num_scheduled_tokens
      insertion order) -- NOT id-sorted (C2 order contract).

    Round-8 verify/commit-side fields (optional maps; a row gains a field
    only when its map carries it -- absence is what the harness's C2-field
    checks report as MISSING_INSTRUMENT):
      - verify[req_id]: {"rej_idx": first rejected position (= num_accepted;
        the bonus position on a full acceptance), "bonus": the target's own
        sampled token at rej_idx (replacement on rejection, bonus on full
        acceptance)} -- both null/absent for rows without a verify window.
      - commit[req_id]: {"emit": tokens committed this step, "stopped":
        whether this commit stopped the request, "phash": committed-prefix
        sha256 after the append, "ol": output-token count,
        "nct": num_computed_tokens} -- the commit/semantic-cache-state
        fingerprint.
      - sampling[req_id]: {"temp","seed","seed_set"} as scheduled (the
        scheduler-side mirror the join cross-checks against the worker's).
    """
    reqs: list[dict] = []
    windows = spec_windows or {}
    counts = acc_rej or {}
    vmap = verify or {}
    cmap = commit or {}
    smap = sampling or {}
    for rid in scheduled_req_ids:
        win = len(windows.get(rid, ()) or ())
        acc, rej = counts.get(rid, (0, 0))
        rq: dict = {
            "id": rid,
            "win": int(win),
            "acc": int(acc),
            "rej": int(rej),
        }
        v = vmap.get(rid)
        if v is not None:
            if v.get("rej_idx") is not None:
                rq["rej_idx"] = int(v["rej_idx"])
            if v.get("bonus") is not None:
                rq["bonus"] = int(v["bonus"])
        c = cmap.get(rid)
        if c is not None:
            rq["emit"] = [int(t) for t in (c.get("emit") or [])]
            if c.get("stopped") is not None:
                rq["stopped"] = bool(c["stopped"])
            if c.get("phash") is not None:
                rq["phash"] = str(c["phash"])
            if c.get("ol") is not None:
                rq["ol"] = int(c["ol"])
            if c.get("nct") is not None:
                rq["nct"] = int(c["nct"])
        s = smap.get(rid)
        if s is not None:
            if s.get("temp") is not None:
                rq["temp"] = float(s["temp"])
            if s.get("seed") is not None:
                rq["seed"] = int(s["seed"])
            if s.get("seed_set") is not None:
                rq["seed_set"] = bool(s["seed_set"])
        reqs.append(rq)
    return {
        "src": source,
        "seq": next_seq(source),
        "t": _now_us(),
        "launch_k": int(launch_k),
        "requested_k": int(requested_k),
        "bs": len(reqs),
        "mode": str(mode),
        "reqs": reqs,
    }


def emit_line(logger, record: dict) -> None:
    """Render one C2 evidence line: the fixed tag + one JSON object."""
    logger.info("%s %s", TAG, json.dumps(record))
