# v20-fp8 MONITORING.md: the R-061 signature-monitoring registration of the promoted line

The promoted fp8-KV serving line (this config directory, the
fp8kv-promo-20261024 image) carries the R-061 signature-monitoring class
registered by DECISIONS.md D-021 item 1 (the byte-exactness demotion) and
consumed by the owner decision OD-4 (D-022) with the fork-rate disclosure
(bf16 0/20 versus fp8 15/20 FORKED on the fixed image, Fisher p=7.7e-07;
the A-12 BRANCH-2 finding, mechanism open). This document registers the
STANDING PER-BOOT MONITOR of that class on the promoted line: every boot
of this config records its signature probe before the boot is put to
work, and the record lands in a dated run directory.

## 1. The standing per-boot signature probe (the embed)

One boot = one monitor record with exactly these parts, in this order:

1. THE T-1a BOOT CANARY (the monitor sequence's first request): the
   registered canary request (b2tripwire CANARY_PROMPT, completions,
   max_tokens 64, temperature 0, seed 7), compared byte-exact against the
   committed inert reference
   (b2/nc/nc2-off/streams/canary.txt, sha256
   5db96eb8309dbe4dbf107e2e2fe4807c6d8444f669307901ec9688b28681d17c).
   Any byte mismatch at a common position FIRES: the corruption-class
   signature, zero tolerance, the boot stops and the fire is reported
   verbatim. A served text shorter than the reference with byte-exact
   prefix agreement is the registered stop-boundary artifact family
   (recorded, not a fire).
2. THE CALIBRATION TRIPLET: the frozen probe bytes of the archived
   sg4-t1b-diff-pre construction (prompt sha256
   d113cace517c4a0c93c38607fa973de32c45aec277c83a9193401c09f654c30d,
   127,495 UTF-8 bytes) served THREE times in-boot under the registered
   request form (completions, max_tokens 96, temperature 0, seed 7,
   top_p 0.95, top_k 20) plus the margin capture (logprobs 5,
   top_logprobs 5; the sampling fields unchanged). All THREE pairs
   compared; any within-boot serving pair divergence at any recorded
   margin = FORKED (the A-11 rule carried by D-021); fork_index is the
   first differing logged-token index (0-based) beside the first
   differing byte offset; the margins of both servings at the fork
   decision are recorded. Per serving the top-two margin of every
   emitted token is captured (the per-serving minimum and mean beside
   the full arrays as side files) and the spec-decode acceptance deltas
   (vllm:spec_decode_num_drafts_total,
   vllm:spec_decode_num_accepted_tokens_total and the per-position
   counter, the prefix-cache counters) are recorded.
3. THE T-1a RE-ISSUE SAMPLING: one canary re-issue after the triplet,
   and one further re-issue after any heavy traffic window (the quality
   suites, a scored campaign, a benchmark session) run on the boot. A
   re-issue fire takes the same zero-tolerance stop as the boot canary.

The driver of record:
agents/glm-5.3-flash-integrator/runs/2026-10-17-qsa-fp8-kv/od4/validation/monitor_triplet.py
(phases boot and post; exit 42 on a T-1a fire, exit 3 on a transport
fault, exit 0 otherwise). The first application of record:
agents/glm-5.3-flash-integrator/runs/2026-10-17-qsa-fp8-kv/od4/validation/
(the OD-4 promotion validation, monitor/r61-embed-boot.json and
monitor/r61-embed-post.json).

## 2. The reduced form, stated exactly

The standing monitor is the REDUCED hook-free form: no sentinel
machinery, no C-GEN reconstruction leg, no in-image capture rows (the
promotion image carries no investigation hooks by construction, the
OD-4 build record). The margin capture is the API-layer target
alternative list returned by the serving endpoint, NOT the A-12
drafter-proposal-logit layer: that instrumentation stays the
investigation-class instrument and runs only on instrumented
investigation images under their own dated registrations. The embed
makes NO protected-surface pass claim and NO calibration-qualification
claim; the A-11 machinery form (b2-probe-protocol-v2.md section 26)
remains the investigation-class probe. A within-boot serving-pair
divergence (a fork) is the REGISTERED monitored phenomenon of this
line: it is recorded verbatim as the signature observation with its
margins and is not by itself a serving stop; the T-1a content tripwire
is the corruption-class detector and its fire IS a stop. The bound
accounting and the disclosure of record (the D-021 campaign, the A-12
finding) carry unchanged.

## 3. The per-boot record convention

- The record lands under
  agents/glm-5.3-flash-integrator/runs/<YYYY-MM-DD>-<slug>/monitor/
  as r61-embed-boot.json (the boot phase) and r61-embed-post.json (the
  post-suites phase), with the per-serving margin side files
  (monitor-margins-f7cal-serveN.json) and the embed stdout logs.
- The verdict fields of record: t1a_fire, triplet_outcome
  (FORKED or UNFORKED), triplet_forked_pairs, per-serving
  text_sha256/prompt_tokens/cached_tokens/finish_reason/margin stats
  and the spec-decode deltas, the fork_decision_margins per forked pair.
- A boot whose monitor record is missing is NOT a validated boot: the
  convention is that the signature probe precedes the boot's workload.

## 4. The class obligations that carry with this registration

- The epsilon 0.001 / delta 0.0005 constants stay frozen (never
  estimated or enlarged from any outcome).
- Any future conjunct assuming byte-identity must first cite platform
  two-arm evidence (the D-021 standing rule).
- The A-12 mechanism question stays open; the intra-kernel probe and
  every other GPU investigation still requires its own dated
  registration before any GPU work.

NO em dash anywhere in this document.
