#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Recompute the layer-3 GEMM1 offline from the shim-captured real input and
compare against the dumped in-serving variants.

Usage: recompute_ref.py /model /detout
"""
import glob
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


def sha_f32(t):
    return hashlib.sha256(
        t.detach().float().contiguous().cpu().numpy().tobytes()
    ).hexdigest()[:16]


def main():
    ckpt, detdir = sys.argv[1], sys.argv[2]
    dev = torch.device("cuda")
    tensors = load_layer3(ckpt)
    weights = build_repacked_weights(tensors, dev)
    w13, w2, w13_s, w2_s = weights

    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new)
    from vllm.scalar_type import scalar_types

    # the layer-3 gemm1 input captured in serving context
    inp = torch.load(f"{detdir}/in_s38x1280x2560_t10_inc7c7bb82f73b.pt")
    x = inp["input"].to(dev).contiguous()
    sorted_ids = inp["sorted"].to(dev)
    expert_ids = inp["expert_ids"].to(dev)
    ntpp = inp["ntpp"].to(dev)
    topk_weights = inp["topk_weights"].to(dev)
    print(f"[in] input {tuple(x.shape)} {x.dtype} sorted_len="
          f"{sorted_ids.shape[0]} ntpp={int(ntpp.item())} "
          f"block={inp['moe_block_size']} mul={inp['mul_topk_weights']} "
          f"ws_at_capture={inp['workspace_hash']}", flush=True)

    workspace = marlin_make_workspace_new(dev, 4)
    workspace.zero_()
    M, N2, K = x.shape[0], 1280, 2560
    cache1 = torch.empty((M * 10, N2), device=dev, dtype=torch.bfloat16)

    ref = ops.moe_wna16_marlin_gemm(
        x, cache1, w13, None, w13_s, None, None, None, None, None,
        workspace, sorted_ids, expert_ids, ntpp, topk_weights,
        moe_block_size=int(inp["moe_block_size"]), top_k=10,
        mul_topk_weights=False, b_q_type=scalar_types.uint4b8,
        size_m=M, size_n=N2, size_k=K, is_k_full=True,
        use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    torch.cuda.synchronize()
    hr, hf = sha_raw(ref), sha_f32(ref)
    print(f"[ref] standalone recompute raw={hr} f32cast={hf}", flush=True)

    # repeat standalone 20x with the SAME real input: does the race fire?
    variants = {}
    for i in range(20):
        out = ops.moe_wna16_marlin_gemm(
            x, cache1, w13, None, w13_s, None, None, None, None, None,
            workspace, sorted_ids, expert_ids, ntpp, topk_weights,
            moe_block_size=int(inp["moe_block_size"]), top_k=10,
            mul_topk_weights=False, b_q_type=scalar_types.uint4b8,
            size_m=M, size_n=N2, size_k=K, is_k_full=True,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        torch.cuda.synchronize()
        variants.setdefault(sha_raw(out), 0)
        variants[sha_raw(out)] += 1
    print(f"[ref] standalone x20 variants: {variants}", flush=True)

    # compare against the dumped in-serving variants
    dumps = {}
    for p in sorted(glob.glob(
            f"{detdir}/dump_s38x1280x2560_t10_inc7c7bb82f73b_*.pt")):
        h = p.rsplit("_", 1)[1][:-3]
        dumps[h] = torch.load(p).view(M * 10, N2)
    print(f"[cmp] {len(dumps)} dumped variants", flush=True)
    res = {"ref_raw": hr, "ref_f32": hf, "standalone_variants": variants,
           "dumped": {}}
    torch.save(ref.cpu(), f"{detdir}/ref_s38x1280x2560_t10.pt")
    for h, t in dumps.items():
        t = t.to(dev)
        same_raw = sha_raw(t)
        diff = (t.view(torch.int16) != ref.view(torch.int16))
        n = int(diff.sum())
        dv = (t.float() - ref.float())[diff] if n else None
        cells = [[int(r), int(c)] for r, c in diff.nonzero().tolist()]
        rec = {
            "raw_hash": same_raw,
            "equals_ref_raw": same_raw == hr,
            "n_diff_halfwords_vs_ref": n,
            "frac": n / t.numel(),
            "max_abs_diff": float(dv.abs().max()) if n else 0.0,
            "cells": cells[:64],
        }
        res["dumped"][h] = rec
        print(f"[cmp] variant {h}: equals_ref={same_raw == hr} "
              f"ndiff={n}/{t.numel()} ({n / t.numel():.2e}) "
              f"maxdiff={rec['max_abs_diff']:.3e} "
              f"cells={cells[:8]}", flush=True)
    json.dump(res, open(f"{detdir}/recompute_ref.json", "w"), indent=1)


if __name__ == "__main__":
    main()
