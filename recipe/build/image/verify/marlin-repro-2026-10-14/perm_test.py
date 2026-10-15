#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Permutation test: does the within-expert-group ORDER of sorted_token_ids
(the moe_align_block_size kernel's atomic-ordering nondeterminism) change the
gemm output at the flip cells?

Runs the layer-3 GEMM1 with the shim-captured real input, permuting the valid
entries inside each expert group of sorted_token_ids with different seeds.
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro import build_repacked_weights, load_layer3  # noqa: E402


def sha_raw(t):
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def main():
    ckpt, detdir = sys.argv[1], sys.argv[2]
    dev = torch.device("cuda")
    tensors = load_layer3(ckpt)
    w13, w2, w13_s, w2_s = build_repacked_weights(tensors, dev)

    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new)
    from vllm.scalar_type import scalar_types

    inp = torch.load(f"{detdir}/in_s38x1280x2560_t10_inc7c7bb82f73b.pt")
    x = inp["input"].to(dev).contiguous()
    sorted0 = inp["sorted"].to(dev)
    expert_ids = inp["expert_ids"].to(dev)
    ntpp = inp["ntpp"].to(dev)
    topk_weights = inp["topk_weights"].to(dev)
    bs = int(inp["moe_block_size"])
    M = x.shape[0]
    SENT = M * 10

    def run(sorted_ids, ws=None):
        if ws is None:
            ws = marlin_make_workspace_new(dev, 4)
            ws.zero_()
        cache1 = torch.empty((M * 10, 1280), device=dev, dtype=torch.bfloat16)
        out = ops.moe_wna16_marlin_gemm(
            x, cache1, w13, None, w13_s, None, None, None, None, None,
            ws, sorted_ids, expert_ids, ntpp, topk_weights,
            moe_block_size=bs, top_k=10, mul_topk_weights=False,
            b_q_type=scalar_types.uint4b8, size_m=M, size_n=1280, size_k=2560,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
            is_zp_float=False)
        torch.cuda.synchronize()
        return out

    ref = run(sorted0)
    h_ref = sha_raw(ref)
    print(f"[perm] reference (captured order): {h_ref}", flush=True)

    # group structure from expert_ids: block b covers sorted[b*8:(b+1)*8]
    nblocks = int(ntpp.item()) // bs
    ei = expert_ids.cpu().tolist()
    groups = []
    i = 0
    while i < nblocks:
        j = i
        while j + 1 < nblocks and ei[j + 1] == ei[i]:
            j += 1
        groups.append((i, j))
        i = j + 1
    print(f"[perm] expert runs: {len(groups)} (first 5: {groups[:5]})",
          flush=True)

    results = {}
    aff_cells = [(155, 582), (198, 382), (198, 552), (198, 1219),
                 (227, 1211), (285, 734)]
    for seed in range(8):
        s = sorted0.clone()
        g = torch.Generator().manual_seed(seed)
        for (a, b) in groups:
            seg = s[a * bs:(b + 1) * bs]
            valid = (seg < SENT).sum().item()
            v = seg[:valid]
            perm = torch.randperm(valid, generator=g)
            s[a * bs:a * bs + valid] = v[perm]
        out = run(s)
        h = sha_raw(out)
        cells = []
        for (r, c) in aff_cells:
            if out[r, c] != ref[r, c]:
                cells.append(((r, c), float(out[r, c]) - float(ref[r, c])))
        same = h == h_ref
        print(f"[perm] seed={seed}: hash={h} same_as_ref={same} "
              f"affcell_diffs={cells}", flush=True)
        results[seed] = {"hash": h, "same": same, "cells": cells}
    json.dump({"ref": h_ref, "results": results},
              open(f"{detdir}/perm_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
