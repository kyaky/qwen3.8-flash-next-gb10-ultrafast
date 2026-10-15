#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P2 stage-1 closure: runtime BEHAVIORAL fingerprints (D-014 condition 1).

Applied ON TOP of the iter7d image state (the moe-align stable-sort shim is
already in marlin_moe.py; model_runner.py carries the stage-1b emitter chain
and the iter7c DETHASH load_model hook). Three fingerprint surfaces, all
under the fixed tag AWFP (one JSON object per line, a "kind" field per
record):

  1. CODE (marlin_moe.py, import time, once per process): the load-time
     code-state check -- the two moe_wna16_marlin_gemm call sites carry
     the atomicAdd tail flag hard-set false (THE proof that atomic_add=0
     is in effect for the Marlin MoE GEMMs: the wrapper's
     should_use_atomic_add_reduce gate can only make the DENSE marlin path
     eligible, and it resolves False unless VLLM_MARLIN_USE_ATOMIC_ADD=1
     AND the shape is eligible -- recorded, never assumed), the stable-sort
     shim's marker and call site are present, and the shim's env-gate state
     is recorded.
  2. PROBE (model_runner.py load_model tail, once per worker process): the
     BEHAVIORAL proof that the shim is active -- a probe call THROUGH the
     patched path with a known input asserts the canonical within-group
     output order, both (i) directly on a crafted non-canonical input and
     (ii) through the REAL composition (moe_align_block_size + the shim,
     exactly the per-forward Marlin MoE sequence). The shim's own debug
     checker (_q38_moe_align_stable_check) is the asserter -- a single
     source of truth for "canonical": sentinel slots unmoved, per-group
     pair multiset preserved, composite keys strictly ascending. A shim
     that is absent, unwired or env-gated OFF fails the probe LOUDLY.
  3. CONFIG + POLICY/DRIFT (awfp_fingerprint.py, wired here): the
     cache/kernel-selection policy record -- the triton autotune
     selections (the fla/GDN kernels are the autotuned population on this
     stack), the force-first-config pinning state, the inductor
     compile-cache entry count (informational; the cache mount persists
     and grows) and the cudagraph configured+captured sizes -- hashed into
     one policy_sha at scheduled steps 1 and 32 and compared against the
     previous boot's persisted record; a mismatch is an AWFP-DRIFT line
     (the autotune boot-lottery that R-029 exposed, reported loudly).
     The harness additionally compares the policy_sha of every boot it
     manages, so drift is detected even without a persistent cache mount.

Every stage is guarded: the fingerprint must never break the boot -- a
failure becomes an AWFP-FAIL line, and an absent AWFP record is
MISSING_INSTRUMENT to the harness (never a silent pass).

Anchored, assert-guarded edits:

  marlin_moe.py
    - appends the qwen38-awfp block AFTER the stable-sort shim block
      (the shim's end marker is a required chain marker: the fingerprint
      is defined against the shimmed file), adding _q38_awfp_code_state()
      and _q38_awfp_align_probe(device) and running the code-state check
      at import.
  gpu/model_runner.py (V2)
    - the load_model tail gains the AWFP install (CONFIG + PROBE) right
      after the DETHASH install block,
    - the scheduled-step gate (requested_k > 0, the same gate the C2
      worker emitter fires under) gains the settled POLICY emit; the
      gate is enforced producer-side inside awfp_fingerprint (the call
      passes this step's requested K and a step without scheduled
      speculative verification neither emits nor counts).

Usage:
    python3 patch_fingerprints.py <marlin_moe.py> <gpu_model_runner.py>
"""
import ast
import sys

if len(sys.argv) != 3:
    print(
        "usage: patch_fingerprints.py <marlin_moe.py> <gpu_model_runner.py>",
        file=sys.stderr,
    )
    sys.exit(64)

MARLIN_PATH = sys.argv[1]
RUNNER_PATH = sys.argv[2]


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
    print(f"patch_fingerprints: {path} OK ({len(edits)} edits, ast.parse pass)")


# ===========================================================================
# 1. marlin_moe.py -- the AWFP code-state check + the behavioral probe
# ===========================================================================
MARLIN_BLOCK = '''

# qwen38-awfp-begin
# --- appended by patch_fingerprints.py (P2 stage-1 closure, D-014 cond. 1) ---
# Runtime behavioral fingerprints for THIS module (marlin_moe), tag AWFP:
#   _q38_awfp_code_state() -- load-time code-state record (pure source
#     reads): the two moe_wna16_marlin_gemm call sites carry the atomicAdd
#     tail flag hard-set false -- THE proof that atomic_add=0 is in effect
#     for the Marlin MoE GEMMs (the drafter's GEMM path); the dense-marlin
#     wrapper's env-conditional gate is a separate, recorded surface. The
#     stable-sort shim's marker and call site must be present and the
#     shim's env-gate state is recorded (a shim-gated-off boot FAILS the
#     probe loudly -- the fingerprint exists to prove activation).
#   _q38_awfp_align_probe(device) -- the BEHAVIORAL canonical-order proof:
#     (i) a crafted NON-canonical within-group order must come back
#     canonical through the shim helper, and (ii) the REAL composition
#     (moe_align_block_size + the shim -- the per-forward Marlin MoE
#     sequence) must end canonical on a known routing input. The shim's
#     own debug checker is the asserter (sentinel slots unmoved, per-group
#     pair multiset preserved, composite keys strictly ascending), so
#     "canonical" has a single source of truth. A shim that is absent,
#     unwired or env-gated off fails the probe loudly.
# The code-state check runs at IMPORT (below); the probe runs once per
# worker process at model-load end (awfp_fingerprint.maybe_install).
# Counting discipline: this block deliberately builds the atomic-flag text
# by concatenation so its own source never adds a literal to the count it
# performs.
import json as _q38_awfp_json


def _q38_awfp_code_state():
    src = open(__file__, "r").read()
    # Counting discipline: every needle is built by concatenation so this
    # block's own source never adds a literal to the counts it performs.
    atomic_text = "use_atomic_" + "add=False"
    shim_marker_text = "# qwen38-moe-align-stable-" + "begin"
    shim_call_text = ("sorted_token_ids = _q38_moe_"
                      "align_stable(")
    return {
        "atomic_add_sites": src.count(atomic_text),
        "shim_marker": src.count(shim_marker_text),
        "shim_call_site": src.count(shim_call_text),
        "shim_env_on": bool(_Q38_MOE_ALIGN_STABLE),
        "file": __file__,
    }


def _q38_awfp_align_probe(device):
    import torch as _q38_awfp_torch
    out = {}
    # (i) DIRECT: a crafted within-group order that is NOT ascending must
    # come back canonical. All ids are valid (none equals the sentinel
    # numel); expert_ids has EXACTLY numel/bs entries (one per block, the
    # real align contract -- two blocks per group here, so multi-block
    # groups are covered too).
    bs = 2
    expert_ids = _q38_awfp_torch.tensor(
        [0, 0, 1, 1, 2, 2], dtype=_q38_awfp_torch.int32, device=device,
    )
    sorted_in = _q38_awfp_torch.tensor(
        [5, 2, 9, 0, 11, 3, 7, 1, 6, 10, 4, 8],
        dtype=_q38_awfp_torch.int32, device=device,
    )
    direct = _q38_moe_align_stable(sorted_in, expert_ids, bs, 12)
    _q38_moe_align_stable_check(sorted_in, direct, expert_ids, bs, 12)
    out["direct"] = "ok"
    # (ii) COMPOSED: the REAL path -- the align kernel the Marlin MoE
    # calls per forward, then the shim. 4 tokens x top-2 over 8 experts.
    topk_ids = _q38_awfp_torch.tensor(
        [[0, 3], [3, 0], [5, 3], [0, 5]],
        dtype=_q38_awfp_torch.int32, device=device,
    )
    s2, e2, n2 = moe_align_block_size(
        topk_ids, bs, 8, None, ignore_invalid_experts=True
    )
    composed = _q38_moe_align_stable(s2, e2, bs, int(topk_ids.numel()))
    _q38_moe_align_stable_check(s2, composed, e2, bs, int(topk_ids.numel()))
    out["composed"] = "ok"
    out["composed_padded"] = int(n2)
    return out


try:
    _Q38_AWFP_CODE = _q38_awfp_code_state()
    _Q38_AWFP_CODE_OK = (
        _Q38_AWFP_CODE["atomic_add_sites"] == 2
        and _Q38_AWFP_CODE["shim_marker"] == 1
        and _Q38_AWFP_CODE["shim_call_site"] == 1
    )
    print(
        "AWFP "
        + _q38_awfp_json.dumps(
            {"kind": "CODE", "ok": bool(_Q38_AWFP_CODE_OK), **_Q38_AWFP_CODE}
        ),
        flush=True,
    )
    if not _Q38_AWFP_CODE_OK:
        print(
            "AWFP-FAIL marlin code-state check failed (see the CODE line): "
            "the atomic_add call-site proof or the stable-sort shim wiring "
            "is broken on this image",
            flush=True,
        )
except Exception as _q38_awfp_exc:  # never break the import
    _Q38_AWFP_CODE_OK = False
    print(
        f"AWFP-FAIL marlin code-state error: {_q38_awfp_exc!r}", flush=True
    )
# qwen38-awfp-end
'''

patch(
    MARLIN_PATH,
    edits=[
        (
            "# qwen38-moe-align-stable-end\n",
            "# qwen38-moe-align-stable-end\n" + MARLIN_BLOCK,
        ),
    ],
    must_contain=[
        "# qwen38-moe-align-stable-end",
        "def _q38_moe_align_stable(",
        "def _q38_moe_align_stable_check(",
    ],
    must_not_contain=["qwen38-awfp-begin"],
)

# ===========================================================================
# 2. gpu/model_runner.py -- the AWFP install (CONFIG + PROBE) at model-load
#    end and the settled POLICY emit at the scheduled-step gate
# ===========================================================================
patch(
    RUNNER_PATH,
    edits=[
        # ---- FP1: the load_model tail, right after the DETHASH install ----
        (
            """        except Exception as _dethash_err:
            logger.error("DETHASH install failed: %s", _dethash_err)
        # ---- end DETHASH ----

    def get_model(self) -> nn.Module:
""",
            """        except Exception as _dethash_err:
            logger.error("DETHASH install failed: %s", _dethash_err)
        # ---- end DETHASH ----

        # ---- AWFP boot fingerprints (P2 stage-1 closure, D-014 cond. 1)
        # ---- _awfp_install
        # Boot-time runtime fingerprints beyond env hashes: AWFP-CONFIG
        # (versions + the kernel-selection env gates + cudagraph/spec
        # config + the marlin code-state) and the BEHAVIORAL probe (the
        # moe-align stable-sort shim's canonical-order proof through the
        # patched path, plus the atomic_add call-site assertion). One-shot
        # per process; every failure is an AWFP-FAIL line, never a boot
        # break. Opt-out: VLLM_AWFP=0 -- an opted-out boot is
        # MISSING_INSTRUMENT to the gate harness (the fingerprint's
        # absence is itself loud).
        try:
            import awfp_fingerprint as _awfp
            _awfp.maybe_install(self)
        except Exception as _awfp_err:
            logger.error("AWFP install failed: %s", _awfp_err)
        # ---- end AWFP install ----

    def get_model(self) -> nn.Module:
""",
        ),
        # ---- FP2: the settled policy fingerprint, producer-gated on the
        # scheduled-step requested K: the call passes the same
        # awdbg_requested_k the C2 worker emitter gates on and the gate
        # lives INSIDE awfp_fingerprint.maybe_emit_policy, so a step
        # without scheduled speculative verification (the JIT warmup
        # steps carry 0, the dummy/profile runs None) neither emits nor
        # advances the emit counter ----
        (
            """            # ---- end GB10 AWDBG1B worker evidence ----
""",
            """            # ---- AWFP settled policy fingerprint (P2 stage-1 closure)
            # ---- _awfp_policy
            # The cache/kernel-selection policy hash (triton autotune
            # selections incl. the fla/GDN kernels, the force-first-config
            # pinning state, the inductor compile-cache entry count,
            # cudagraph configured+captured sizes) emitted at scheduled
            # steps 1 and 32: the first scheduled step runs after the JIT
            # warmup (the selections are settled by then); the step-32
            # record catches late cudagraph captures. PRODUCER-GATED: the
            # step's requested K rides along and the gate lives INSIDE
            # awfp_fingerprint.maybe_emit_policy -- a step with no
            # speculative verification scheduled neither emits nor
            # advances the emit counter, so the counter counts SCHEDULED
            # steps only and stays aligned with the C2 worker emitter's
            # gate even if this call site moves. Cross-boot drift
            # (the autotune boot-lottery, R-029) is an AWFP-DRIFT line
            # here and a harness-level comparison across the boots it
            # manages. One-shot flags live inside awfp_fingerprint;
            # failures are AWFP-FAIL lines, never boot breaks.
            try:
                import awfp_fingerprint as _awfp_pol
                _awfp_pol.maybe_emit_policy(self, awdbg_requested_k)
            except Exception as _awfp_pol_err:
                logger.error("AWFP policy emit failed: %s", _awfp_pol_err)
            # ---- end GB10 AWDBG1B worker evidence ----
""",
        ),
    ],
    must_contain=[
        "_awdbg_worker_emitter",
        "end DETHASH ----",
    ],
    must_not_contain=[
        "_awfp_install",
        "_awfp_policy",
    ],
)

print("patch_fingerprints: both files patched OK")
