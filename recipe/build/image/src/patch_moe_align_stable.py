#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Canonical within-group order for moe_align_block_size (D-011 tie-flip fix).

Root cause (causally proven: runs/2026-10-14-det-localize + 2026-10-14-marlin-
repro): vllm's moe_align_block_size scatters each (token, top-k) pair into its
expert group's slot range with an atomicAdd on a per-expert cursor
(csrc/moe/moe_align_sum_kernels.cu, _count_and_sort_expert_tokens:341-368,
verified verbatim in the serving image), so the order of pairs INSIDE one
expert group depends on thread arrival order and varies between bit-identical
forwards. The moe_wna16 Marlin kernel is sensitive to that cross-block pair
placement (k-slice split): bf16-boundary cells flip by 1 ULP, which breaks
byte-identical forwards and surfaces as the tie-flip on near-tie top-2 logprobs.

Fix (this patch, env-gated VLLM_MOE_ALIGN_STABLESORT=1): at the call site in
fused_marlin_moe (the per-forward Marlin MoE path), right after
moe_align_block_size returns (sorted_token_ids, expert_ids,
num_tokens_post_padded), sort each expert group's contiguous segment of
sorted_token_ids ascending by pair id. Deterministic and semantically
identical: the kernel pre-fills the whole buffer with the sentinel value
numel and scatters each group's valid pair ids (all < numel) into the group's
leading slots, so an ascending run-local sort only reorders the valid ids
inside their existing slot range; blocks past the padded length hold -1 in
expert_ids and sentinel values in sorted_token_ids (their own terminal run).
Group membership, per-block expert ids and the padded length are unchanged.
The per-group pair SET is deterministic; only arrival order varies.

Scope notes:
  - The batched align kernel (batched_moe_align_block_size) inserts
    sequentially per batch - no atomicAdd, already deterministic - so the
    batched Marlin path needs no shim.
  - The LoRA align path (moe_lora_align_block_size) shares the atomicAdd
    scatter but is not exercised by this serving stack (no LoRA adapters);
    left untouched.
  - Upstream the real fix is a one-line deterministic scatter in
    moe_align_block_size (ordered insertion instead of the atomicAdd); this
    python shim is the local fix, the native fix is the upstream proposal.

Usage: patch_moe_align_stable.py <site-packages-dir>
Idempotent: re-running on a patched tree is a verified no-op.
"""
import ast
import sys

sp = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages"
MM = f"{sp}/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"

CALL = """    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )
"""
CALL_PATCHED = CALL + """    # qwen38-moe-align-stable: canonical within-group order (D-011 tie-flip fix)
    sorted_token_ids = _q38_moe_align_stable(
        sorted_token_ids, expert_ids, block_size_m, topk_ids.numel()
    )
"""

MARK_BEGIN = "# qwen38-moe-align-stable-begin"
MARK_END = "# qwen38-moe-align-stable-end"

HELPER = f'''

{MARK_BEGIN}
# --- appended by patch_moe_align_stable.py (VLLM_MOE_ALIGN_STABLESORT=1) ---
# Canonical within-group order for the Marlin fused-MoE path (D-011):
# moe_align_block_size scatters each expert group's (token, top-k) pairs with
# an atomicAdd on a per-expert cursor (csrc/moe/moe_align_sum_kernels.cu,
# _count_and_sort_expert_tokens), so the order of pairs INSIDE one expert
# group depends on thread arrival order and varies between bit-identical
# forwards. The moe_wna16 Marlin kernel is sensitive to that cross-block pair
# placement (k-slice split): bf16-boundary cells flip by 1 ULP and
# byte-identical forwards break. Sorting each expert group's contiguous
# segment of sorted_token_ids ascending by pair id makes the order canonical:
# the align kernel pre-fills the whole buffer with the sentinel value numel
# and writes each group's valid pair ids (all < numel) into the group's
# leading slots, and blocks past the padded length hold -1 in expert_ids with
# sentinel values in sorted_token_ids, so an ascending run-local sort only
# reorders valid ids inside their existing slot range. Group membership,
# per-block expert ids and the padded length are unchanged. The per-group
# pair SET is deterministic; only arrival order varies. Upstream the fix is a
# one-line deterministic scatter (ordered insertion instead of the atomicAdd);
# this python shim is the local fix, the native fix is the upstream proposal.
import os as _q38_mas_os

_Q38_MOE_ALIGN_STABLE = (
    _q38_mas_os.environ.get("VLLM_MOE_ALIGN_STABLESORT", "0") == "1"
)
_Q38_MOE_ALIGN_STABLE_DEBUG = (
    _q38_mas_os.environ.get("VLLM_MOE_ALIGN_STABLESORT_DEBUG", "0") == "1"
)


def _q38_moe_align_stable(sorted_token_ids, expert_ids, block_size, numel):
    """Sort each expert group's segment of sorted_token_ids ascending.

    Deterministic on identical input and semantically identical to the raw
    align output for every consumer of (sorted_token_ids, expert_ids,
    num_tokens_post_padded). Pure GPU tensor ops, no host sync, so it is safe
    inside piecewise CUDA-graph capture and replays deterministically.
    """
    if not _Q38_MOE_ALIGN_STABLE or sorted_token_ids.numel() == 0:
        return sorted_token_ids
    device = sorted_token_ids.device
    bs = int(block_size)
    numel = int(numel)
    eid = expert_ids
    if eid.dtype != torch.int32:
        eid = eid.to(torch.int32)
    nb = eid.numel()
    assert nb * bs >= sorted_token_ids.numel(), (
        "expert_ids/smaller than sorted_token_ids blocks"
    )
    # Position-ordered group ids: contiguous blocks with the same expert id
    # form one group. The -1 tail past the padded length is its own terminal
    # group (sentinel-only) and never merges with a valid group.
    if nb > 1:
        starts = torch.cat(
            (
                torch.ones((1,), dtype=torch.int32, device=device),
                (eid[1:] != eid[:-1]).to(torch.int32),
            )
        )
    else:
        starts = torch.ones((1,), dtype=torch.int32, device=device)
    run = torch.cumsum(starts, 0).repeat_interleave(bs)
    run = run[: sorted_token_ids.numel()]
    # Composite key: group-major, pair-id-minor. Valid ids are < numel and the
    # sentinel is exactly numel, so sentinels sort after the valid ids of
    # their group; equal sentinels are interchangeable. int32 keys are exact
    # whenever the largest key fits, which covers every serving shape.
    if nb * (numel + 1) + numel < 2**31:
        key = run * (numel + 1) + sorted_token_ids
    else:
        key = run.to(torch.int64) * (numel + 1) + sorted_token_ids.to(
            torch.int64
        )
    out = sorted_token_ids[torch.argsort(key, stable=True)]
    if _Q38_MOE_ALIGN_STABLE_DEBUG:
        _q38_moe_align_stable_check(
            sorted_token_ids, out, expert_ids, bs, numel
        )
    return out


def _q38_moe_align_stable_check(before, after, expert_ids, bs, numel):
    """Debug guard (host-syncing; VLLM_MOE_ALIGN_STABLESORT_DEBUG=1 only)."""
    assert before.shape == after.shape, "shape changed"
    assert before.dtype == after.dtype, "dtype changed"
    vb = before != numel
    assert torch.equal(vb, after != numel), "sentinel slots moved"
    assert torch.equal(
        before[vb].sort().values, after[vb].sort().values
    ), "valid pair multiset changed"
    nb = expert_ids.numel()
    eid = expert_ids.to(torch.int64)
    starts = torch.ones((nb,), dtype=torch.int64, device=before.device)
    if nb > 1:
        starts[1:] = (eid[1:] != eid[:-1]).to(torch.int64)
    run = torch.cumsum(starts, 0).repeat_interleave(bs)
    run = run[: before.numel()]
    # Per-group membership preserved (the composite key is injective per
    # (group, pair id), so equal sorted keys mean equal per-group multisets).
    kb = (run * (numel + 1) + before.to(torch.int64))[vb]
    ka = (run * (numel + 1) + after.to(torch.int64))[vb]
    assert torch.equal(kb.sort().values, ka.sort().values), (
        "per-group pair membership changed"
    )
    # ... and the after side is canonical: composite keys strictly ascend
    # across the whole valid sequence (groups are value-ordered run-major,
    # and pair ids are unique inside a group).
    kraw = ka
    assert kraw.numel() < 2 or bool((kraw[1:] > kraw[:-1]).all()), (
        "within-group order not canonical"
    )
{MARK_END}
'''

with open(MM) as f:
    s = f.read()

if MARK_BEGIN in s:
    assert s.count(MARK_BEGIN) == 1 and s.count(MARK_END) == 1, (
        "marker count mismatch on already-patched file"
    )
    assert CALL_PATCHED in s, "patched call site missing on already-patched file"
    assert "_q38_moe_align_stable" in s
    ast.parse(s)
    print("moe-align-stable: already applied (verified, no-op)")
    sys.exit(0)

assert "\nimport torch\n" in s, "marlin_moe.py does not import torch"
assert s.count(CALL) == 1, (
    f"moe_align_block_size call site: expected 1, found {s.count(CALL)}"
)
s = s.replace(CALL, CALL_PATCHED)
s += HELPER
ast.parse(s)
with open(MM, "w") as f:
    f.write(s)
print("marlin_moe.py: moe-align-stable canonicalization applied OK")
