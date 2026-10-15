#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU self-check for the moe-align stable-sort shim (D-011 tie-flip fix).

Runs the REAL vllm moe_align_block_size wrapper (the exact call the Marlin
fused-MoE path makes per forward) on synthetic topk_ids, canonicalizes each
output with the shipped helper (extracted from the patched marlin_moe.py
marker block, debug asserts armed), and verifies:

  1. grouping is deterministic across identical calls (expert_ids and
     num_tokens_post_padded bit-identical);
  2. the canonicalized output is bit-identical across identical calls even
     when the raw align output is not (the determinism fix itself);
  3. every valid entry sits in a block owned by its pair's expert in
     topk_ids (ground-truth membership, the property the Marlin kernel needs);
  4. within each expert group the valid pair ids ascend (canonical order);
  5. sentinel slots (== numel) keep their positions; valid multiset preserved;
  6. byte-equal to the round-5 reference shim; idempotent;
  7. benchmark: GPU composite-key argsort vs the round-5 CPU-loop shim.

usage: test_moe_align_stable_gpu.py [path-to-marlin_moe.py]
"""
import hashlib
import os
import random
import sys
import time

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

mm_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/experts/marlin_moe.py"
)
os.environ["VLLM_MOE_ALIGN_STABLESORT"] = "1"
os.environ["VLLM_MOE_ALIGN_STABLESORT_DEBUG"] = "1"
src = open(mm_path).read()
block = src[src.index("# qwen38-moe-align-stable-begin"):src.index(
    "# qwen38-moe-align-stable-end")]


def load_helper():
    ns = {"torch": torch}
    exec(compile(block, mm_path, "exec"), ns)  # noqa: S102 - repo-owned source
    return ns["_q38_moe_align_stable"]


canon = load_helper()  # debug asserts armed for the correctness phase


def reference(sorted_ids, expert_ids, bs, numel):
    """Round-5 measured shim (CPU loop), generalized to any numel."""
    ei = expert_ids.tolist()
    h = sorted_ids.tolist()
    out = list(h)
    i = 0
    nb = len(ei)
    while i < nb:
        j = i
        while j + 1 < nb and ei[j + 1] == ei[i]:
            j += 1
        seg = h[i * bs : (j + 1) * bs]
        v = sum(1 for x in seg if x < numel)
        if v > 1:
            out[i * bs : i * bs + v] = sorted(seg[:v])
        i = j + 1
    return torch.tensor(out, dtype=sorted_ids.dtype, device=sorted_ids.device)


def sha(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:12]


E = 512  # qwen3.8-flash-next routed experts
TOPK = 10
K = 12  # repeated identical align calls
rng = random.Random(311)

# (M, block_size_m) following fused_marlin_moe's block-size selection:
# smallest bs in [8,16,32,48,64] with M*topk/E/bs < 0.9.
SHAPES = [(24, 8), (256, 8), (1024, 32), (4096, 64)]
all_ok = True
for M, bs in SHAPES:
    topk_ids = torch.stack(
        [
            torch.randperm(E, device="cuda")[:TOPK]
            for _ in range(M)
        ]
    ).contiguous()
    # make ties likely: heavy reuse of a few experts
    hot = torch.tensor([7, 19, 33, 101, 256, 511], device="cuda")
    mask = torch.rand(M, TOPK, device="cuda") < 0.5
    topk_ids = torch.where(
        mask, hot[torch.randint(0, 6, (M, TOPK), device="cuda")], topk_ids
    )
    numel = topk_ids.numel()

    raws, cans = [], []
    expert_ids0 = ntp0 = None
    for it in range(K):
        s, e, n = moe_align_block_size(
            topk_ids, bs, E, None, ignore_invalid_experts=True
        )
        if it == 0:
            expert_ids0, ntp0 = e, int(n.item())
        assert torch.equal(e, expert_ids0), "expert_ids not deterministic"
        assert int(n.item()) == ntp0, "num_tokens_post_padded not deterministic"
        c = canon(s, e, bs, numel)
        raws.append(sha(s))
        cans.append(sha(c))
        # ground-truth membership: the entry at position p holds pair v and
        # must sit in a block owned by pair v's expert in topk_ids
        topk_flat = topk_ids.view(-1).to(torch.int32)
        valid = c != numel
        pos = valid.nonzero(as_tuple=True)[0]
        got = e[pos // bs]
        want = topk_flat[c[pos]]
        assert torch.equal(got, want), "pair placed in a foreign expert block"
        # canonical order within each group + reference equality + idempotence
        assert torch.equal(c, reference(s, e, bs, numel)), "reference mismatch"
        assert torch.equal(c, canon(c, e, bs, numel)), "not idempotent"
        del s, c

    raw_variants = len(set(raws))
    can_variants = len(set(cans))
    print(
        f"M={M:5d} bs={bs:2d} numel={numel:6d} padded={ntp0:6d} "
        f"raw variants={raw_variants}/{K} canonical variants={can_variants}/{K}"
    )
    if can_variants != 1:
        all_ok = False
        print("  FAIL: canonical output still varies across identical calls")

# ---- benchmark: composite-key argsort vs round-5 CPU-loop shim ----------
# Reload the helper with debug asserts off - the debug guard host-syncs and
# is illegal during CUDA-graph capture (and would inflate every timing).
os.environ["VLLM_MOE_ALIGN_STABLESORT_DEBUG"] = "0"
canon = load_helper()
print("benchmark (per align call, 50 iters after 10 warmup):")
for M, bs in SHAPES:
    topk_ids = torch.randint(0, E, (M, TOPK), device="cuda")
    numel = topk_ids.numel()
    s, e, n = moe_align_block_size(
        topk_ids, bs, E, None, ignore_invalid_experts=True
    )
    for _ in range(10):
        canon(s, e, bs, numel)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        canon(s, e, bs, numel)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / 50 * 1e3
    reference(s, e, bs, numel)  # warm
    t0 = time.perf_counter()
    for _ in range(50):
        reference(s, e, bs, numel)
    cpu_ms = (time.perf_counter() - t0) / 50 * 1e3
    print(
        f"  M={M:5d}: argsort-shim {gpu_ms:7.3f} ms   "
        f"cpu-loop-shim {cpu_ms:7.3f} ms   (x48 layers: "
        f"{gpu_ms * 48:6.2f} ms / fwd)"
    )

assert all_ok, "canonical output varied across identical calls"

# ---- serving-relevant cost: CUDA-graph replay (v17a runs PIECEWISE graphs,
# so python dispatch happens once at capture; replay pays only GPU kernels).
print("cuda-graph replay benchmark (48 align calls per forward):")
for M, bs in SHAPES[:2]:
    topk_ids = torch.randint(0, E, (M, TOPK), device="cuda")
    numel = topk_ids.numel()
    s, e, n = moe_align_block_size(
        topk_ids, bs, E, None, ignore_invalid_experts=True
    )
    g = torch.cuda.CUDAGraph()
    s2, e2 = s.clone(), e.clone()
    for _ in range(10):
        canon(s2, e2, bs, numel)  # warm the allocator paths
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(48):
            canon(s2, e2, bs, numel)
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    per_fwd = (time.perf_counter() - t0) / 100 * 1e3
    print(f"  M={M:5d}: captured 48 canonicalizations, replay {per_fwd:6.3f} ms/forward")

print("test_moe_align_stable_gpu: PASS")
