#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D-011 follow-up: faithful standalone repro of ops.moe_wna16_marlin_gemm
bimodality (layer-3 GEMM1 M=24 N=1280 K=2560 top_k=10; GEMM2 M=240 N=2560
K=640 top_k=1) on the iter7c-20261014 image.

Faithful construction:
- REAL layer-3 expert weights from the checkpoint (gate/up/down qweight,
  qzeros, scales), fused + repacked through the exact loader helpers
  (ops.gptq_marlin_moe_repack / marlin_moe_permute_scales /
  marlin_moe_padded_intermediate) with the same args _process_weights_marlin
  passes (int_wna16.py:533-700).
- EXACT call context of marlin_moe.py:_fused_marlin_moe (lines 135-229):
  use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
  b_q_type=uint4b8 (sym gptq), mul_topk_weights False/True,
  moe_block_size=8 (block_size_m selection loop on M=24 topk=10 E=512),
  workspace=marlin_make_workspace_new(device, 4) created once and reused,
  intermediate caches allocated once (torch.empty) and reused across calls,
  GEMM2 output into the cache13 region then torch.sum over topk.
- Real routing: layer-3 gate.weight -> softmax -> top10 -> renormalized
  fp32 weights, computed ONCE from a fixed seed and reused (identical
  inputs at the op boundary, matching serving where the router is a pure
  function of identical hidden states).

Run inside a throwaway --gpus all container from the iter7c image.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time

import torch


def sha(t: torch.Tensor) -> str:
    b = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def assert_env():
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"[env] device={torch.cuda.get_device_name(dev)} "
          f"cc={torch.cuda.get_device_capability(dev)} sms={props.multi_processor_count}",
          flush=True)
    return dev


def load_layer3(ckpt: str):
    from safetensors import safe_open
    files = ["model-00004-of-00067.safetensors",
             "model-00005-of-00067.safetensors",
             "model-00006-of-00067.safetensors"]
    want_mlp = ".layers.3.mlp"
    tensors = {}
    for f in files:
        with safe_open(f"{ckpt}/{f}", framework="pt") as fh:
            for k in fh.keys():
                if want_mlp in k and ("experts." in k
                                      or k.endswith("mlp.gate.weight")):
                    tensors[k] = fh.get_tensor(k)
    n_exp = len({k.split(".experts.")[1].split(".")[0]
                 for k in tensors if ".experts." in k})
    assert n_exp == 512, f"expected 512 experts, got {n_exp}"
    print(f"[load] layer-3 expert tensors: {len(tensors)}", flush=True)
    return tensors


def build_repacked_weights(tensors, dev, group_size=128):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    E, K, N = 512, 2560, 640
    pack_factor = 8  # 4-bit
    pfx = "model.language_model.layers.3.mlp.experts"

    def stack(name):
        return torch.stack([tensors[f"{pfx}.{e}.{name}"] for e in range(E)]).to(dev)

    gate_qw, up_qw, down_qw = stack("gate_proj.qweight"), stack("up_proj.qweight"), stack("down_proj.qweight")
    gate_sc, up_sc, down_sc = stack("gate_proj.scales"), stack("up_proj.scales"), stack("down_proj.scales")
    assert gate_qw.shape == (E, K // pack_factor, N), gate_qw.shape
    assert down_qw.shape == (E, N // pack_factor, K), down_qw.shape

    qz = stack("gate_proj.qzeros")[:8]
    # sym gptq: qzeros carry a constant filler (AutoRound writes 0x77777777,
    # i.e. nibble 7) and the loader never creates the zp param for sym=True
    # (auto_gptq.py TYPE_MAP (4, True) -> uint4b8; may_have_zp=not is_sym).
    constant = bool((qz == qz.flatten()[0]).all())
    filler = int(qz.flatten()[0].item())
    print(f"[load] qzeros constant filler={filler} (sym -> dropped): "
          f"{constant}; gate_qw {tuple(gate_qw.shape)} {gate_qw.dtype}, "
          f"scales {tuple(gate_sc.shape)} {gate_sc.dtype}", flush=True)
    assert constant, "qzeros not constant -> asymmetric path, adjust repro"

    # w13 = [gate; up] along the output dim (routed_experts.py:511-514);
    # gptq qweight/scales are [K/pf, N] so the output dim is axis 2
    w13_qw = torch.cat([gate_qw, up_qw], dim=2).contiguous()
    # loader keeps scales in params_dtype (bf16 for this model)
    w13_sc = torch.cat([gate_sc, up_sc], dim=2).contiguous().to(torch.bfloat16)
    down_sc = down_sc.to(torch.bfloat16)

    padded_N = marlin_utils.marlin_moe_padded_intermediate(N, group_size)
    assert padded_N == N, f"unexpected padding: {N} -> {padded_N}"

    perm = torch.empty((E, 0), dtype=torch.int32, device=dev)  # desc_act=false
    w13_marlin = ops.gptq_marlin_moe_repack(w13_qw, perm, K, 2 * N, 4, False)
    w2_marlin = ops.gptq_marlin_moe_repack(down_qw, perm, N, K, 4, False)
    w13_scales = marlin_utils.marlin_moe_permute_scales(
        s=w13_sc, size_k=N, size_n=w13_sc.shape[2],
        group_size=group_size, is_a_8bit=False)
    w2_scales = marlin_utils.marlin_moe_permute_scales(
        s=down_sc, size_k=down_sc.shape[1] * group_size,
        size_n=down_sc.shape[2], group_size=group_size, is_a_8bit=False)
    assert w13_marlin.shape == (E, K // 16, 2 * N * 2), w13_marlin.shape
    assert w2_marlin.shape == (E, N // 16, K * 2), w2_marlin.shape
    assert w13_scales.shape == (E, K // group_size, 2 * N), w13_scales.shape
    assert w2_scales.shape == (E, N // group_size, K), w2_scales.shape
    print(f"[repack] w13 {tuple(w13_marlin.shape)} w2 {tuple(w2_marlin.shape)} "
          f"w13_sc {tuple(w13_scales.shape)} w2_sc {tuple(w2_scales.shape)}",
          flush=True)
    return w13_marlin, w2_marlin, w13_scales, w2_scales


def build_routing(tensors, dev, m=24, top_k=10, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn((m, 2560), generator=g, dtype=torch.float32,
                         device="cpu").to(torch.bfloat16).to(dev)
    gate = tensors["model.language_model.layers.3.mlp.gate.weight"].to(dev)
    scores = torch.softmax(hidden.to(torch.bfloat16) @ gate.t(), dim=-1)
    tw, ti = torch.topk(scores, top_k, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    topk_ids = ti.to(torch.int32)
    topk_weights = tw.to(torch.float32).contiguous()
    assert topk_ids.shape == (m, top_k)
    assert topk_weights.dtype == torch.float32
    return hidden, topk_ids, topk_weights


def build_call_ctx(args, tensors, dev):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new)

    M, top_k, K, N = args.m, args.top_k, 2560, 640
    hidden, topk_ids, topk_weights = build_routing(
        tensors, dev, m=M, top_k=top_k, seed=args.seed)

    # block_size_m selection loop from fused_marlin_moe (marlin_moe.py:333)
    E = 512
    block_size_m = 64
    for b in [8, 16, 32, 48, 64]:
        block_size_m = b
        if M * top_k / E / b < 0.9:
            break
    if args.force_block_size:
        block_size_m = args.force_block_size
    print(f"[ctx] block_size_m={block_size_m}", flush=True)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, E, None, ignore_invalid_experts=True)
    print(f"[ctx] sorted_ids len={sorted_token_ids.shape[0]} "
          f"num_tokens_post_padded={int(num_tokens_post_padded.item())} "
          f"sorted_sha={sha(sorted_token_ids)} "
          f"expert_ids_sha={sha(expert_ids)}", flush=True)

    # one persistent workspace, as serving does (auto_gptq.py:677)
    workspace = marlin_make_workspace_new(dev, 4)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    assert workspace.numel() == sms * 4, (workspace.numel(), sms)

    # intermediate caches: provisioned once, reused every call. Standard
    # format workspace shapes (MarlinExperts.workspace_shapes):
    #   cache13 = (M*topk*max(2N, K),)   cache2 = (M*topk, max(N, K))
    ctx = dict(
        hidden=hidden, topk_ids=topk_ids, topk_weights=topk_weights,
        workspace=workspace, block_size_m=block_size_m, sms=sms,
        sorted=sorted_token_ids, expert_ids=expert_ids,
        ntpp=num_tokens_post_padded, m=M, top_k=top_k,
    )
    alloc_caches(ctx, args.fresh_caches)
    return ctx


def alloc_caches(ctx, keep_sizes):
    M, top_k, K, N = ctx["m"], ctx["top_k"], 2560, 640
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    ctx["cache13_flat"] = torch.empty((M * top_k * max(2 * N, K),),
                                      device="cuda", dtype=torch.bfloat16)
    ctx["cache2_full"] = torch.empty((M * top_k, max(N, K)),
                                     device="cuda", dtype=torch.bfloat16)
    ctx["cache1"] = _resize_cache(ctx["cache13_flat"], (M * top_k, 2 * N))
    ctx["cache3"] = _resize_cache(ctx["cache13_flat"], (M * top_k, K))
    ctx["cache2"] = _resize_cache(ctx["cache2_full"], (M * top_k, N))
    ctx["output_buf"] = torch.empty((M, K), device="cuda",
                                    dtype=torch.bfloat16)


def reset_caches(ctx):
    M, top_k, K, N = ctx["m"], ctx["top_k"], 2560, 640
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    ctx["cache13_flat"] = torch.empty((M * top_k * max(2 * N, K),),
                                      device="cuda", dtype=torch.bfloat16)
    ctx["cache2_full"] = torch.empty((M * top_k, max(N, K)),
                                     device="cuda", dtype=torch.bfloat16)
    ctx["cache1"] = _resize_cache(ctx["cache13_flat"], (M * top_k, 2 * N))
    ctx["cache3"] = _resize_cache(ctx["cache13_flat"], (M * top_k, K))
    ctx["cache2"] = _resize_cache(ctx["cache2_full"], (M * top_k, N))
    ctx["output_buf"] = torch.empty((M, K), device="cuda",
                                    dtype=torch.bfloat16)


def run_pair(args, ctx, weights, stage):
    """GEMM1 -> silu_and_mul -> GEMM2 -> sum, exactly marlin_moe.py:135-231."""
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types

    w13, w2, w13_scales, w2_scales = weights
    M, top_k, K, N = ctx["m"], ctx["top_k"], 2560, 640
    workspace = ctx["workspace"]
    if args.zero_workspace:
        workspace.zero_()
    thread_k = args.thread_k or -1
    thread_n = args.thread_n or -1
    bps = args.blocks_per_sm or -1
    use_fp32_reduce = not args.no_fp32_reduce
    use_atomic_add = args.atomic_add
    b_q_type = scalar_types.uint4b8

    cache1 = ops.moe_wna16_marlin_gemm(
        ctx["hidden"], ctx["cache1"], w13, None, w13_scales, None, None,
        None, None, None, workspace, ctx["sorted"], ctx["expert_ids"],
        ctx["ntpp"], ctx["topk_weights"],
        moe_block_size=ctx["block_size_m"], top_k=top_k,
        mul_topk_weights=False, b_q_type=b_q_type,
        size_m=M, size_n=2 * N, size_k=K,
        is_k_full=True, use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce, is_zp_float=False,
        thread_k=thread_k, thread_n=thread_n, blocks_per_sm=bps)
    assert cache1.data_ptr() == ctx["cache1"].data_ptr()

    torch.ops._C.silu_and_mul(ctx["cache2"], cache1)

    cache3 = ops.moe_wna16_marlin_gemm(
        ctx["cache2"], ctx["cache3"], w2, None, w2_scales, None, None,
        None, None, None, workspace, ctx["sorted"], ctx["expert_ids"],
        ctx["ntpp"], ctx["topk_weights"],
        moe_block_size=ctx["block_size_m"], top_k=1,
        mul_topk_weights=True, b_q_type=b_q_type,
        size_m=M * top_k, size_n=K, size_k=N,
        is_k_full=True, use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce, is_zp_float=False,
        thread_k=thread_k, thread_n=thread_n, blocks_per_sm=bps)
    assert cache3.data_ptr() == ctx["cache3"].data_ptr()

    out = ctx["output_buf"]
    torch.sum(cache3.view(-1, top_k, K), dim=1, out=out)
    hashes = {
        "gemm1": sha(cache1),
        "gemm2": sha(cache3),
        "final": sha(out),
    }
    return hashes


def launch_pair(ctx, weights, tk, tn, bps, ufa, uaa, zero_ws=False):
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types
    w13, w2, w13_scales, w2_scales = weights
    M, top_k, K, N = ctx["m"], ctx["top_k"], 2560, 640
    if zero_ws:
        ctx["workspace"].zero_()
    cache1 = ops.moe_wna16_marlin_gemm(
        ctx["hidden"], ctx["cache1"], w13, None, w13_scales, None, None,
        None, None, None, ctx["workspace"], ctx["sorted"], ctx["expert_ids"],
        ctx["ntpp"], ctx["topk_weights"],
        moe_block_size=ctx["block_size_m"], top_k=top_k,
        mul_topk_weights=False, b_q_type=scalar_types.uint4b8,
        size_m=M, size_n=2 * N, size_k=K, is_k_full=True,
        use_atomic_add=uaa, use_fp32_reduce=ufa, is_zp_float=False,
        thread_k=tk, thread_n=tn, blocks_per_sm=bps)
    torch.ops._C.silu_and_mul(ctx["cache2"], cache1)
    cache3 = ops.moe_wna16_marlin_gemm(
        ctx["cache2"], ctx["cache3"], w2, None, w2_scales, None, None,
        None, None, None, ctx["workspace"], ctx["sorted"], ctx["expert_ids"],
        ctx["ntpp"], ctx["topk_weights"],
        moe_block_size=ctx["block_size_m"], top_k=1,
        mul_topk_weights=True, b_q_type=scalar_types.uint4b8,
        size_m=M * top_k, size_n=K, size_k=N, is_k_full=True,
        use_atomic_add=uaa, use_fp32_reduce=ufa, is_zp_float=False,
        thread_k=tk, thread_n=tn, blocks_per_sm=bps)
    return cache1, cache3


def tight_run(args, ctx, weights, iters, collect_all=True):
    """No per-iter host sync; rotating snapshots; hash after final sync."""
    tk = args.thread_k or -1
    tn = args.thread_n or -1
    bps = args.blocks_per_sm or -1
    ufa = not args.no_fp32_reduce
    uaa = args.atomic_add
    P = iters  # snapshot every iteration (lossless)
    snap1 = [torch.empty_like(ctx["cache1"]) for _ in range(P)]
    snap2 = [torch.empty_like(ctx["cache3"]) for _ in range(P)]
    ws0 = sha(ctx["workspace"])
    for it in range(iters):
        cache1, cache3 = launch_pair(ctx, weights, tk, tn, bps, ufa, uaa,
                                     zero_ws=args.zero_workspace)
        snap1[it].copy_(cache1, non_blocking=True)
        snap2[it].copy_(cache3, non_blocking=True)
    torch.cuda.synchronize()
    ws1 = sha(ctx["workspace"])
    variants = {"gemm1": {}, "gemm2": {}}
    recs = []
    for it in range(iters):
        h1, h2 = sha(snap1[it]), sha(snap2[it])
        variants["gemm1"].setdefault(h1, 0)
        variants["gemm1"][h1] += 1
        variants["gemm2"].setdefault(h2, 0)
        variants["gemm2"][h2] += 1
        if collect_all or variants["gemm1"][h1] > 1:
            recs.append({"i": it, "gemm1": h1, "gemm2": h2})
    return variants, recs, (ws0, ws1)


def run_seed_sweep(args, tensors, weights, dev):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    results = {}
    for seed in range(args.sweep_seeds):
        hidden, topk_ids, topk_weights = build_routing(
            tensors, dev, m=args.m, top_k=args.top_k, seed=seed)
        sorted_ids, expert_ids, ntpp = moe_align_block_size(
            topk_ids, 8, 512, None, ignore_invalid_experts=True)
        ctx = dict(hidden=hidden, topk_ids=topk_ids,
                   topk_weights=topk_weights, sorted=sorted_ids,
                   expert_ids=expert_ids, ntpp=ntpp,
                   block_size_m=args.force_block_size or 8,
                   m=args.m, top_k=args.top_k)
        # fresh workspace + caches per seed (like a fresh layer instance)
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new)
        ctx["workspace"] = marlin_make_workspace_new(dev, 4)
        M, top_k, K, N = args.m, args.top_k, 2560, 640
        from vllm.model_executor.layers.fused_moe.utils import _resize_cache
        cache13 = torch.empty((M * top_k * max(2 * N, K),), device=dev,
                              dtype=torch.bfloat16)
        cache2f = torch.empty((M * top_k, max(N, K)), device=dev,
                              dtype=torch.bfloat16)
        ctx["cache1"] = _resize_cache(cache13, (M * top_k, 2 * N))
        ctx["cache3"] = _resize_cache(cache13, (M * top_k, K))
        ctx["cache2"] = _resize_cache(cache2f, (M * top_k, N))
        variants, recs, ws = tight_run(args, ctx, weights, 200)
        n1 = len(variants["gemm1"])
        n2 = len(variants["gemm2"])
        print(f"[sweep] seed={seed} nttp={int(ntpp.item())} "
              f"gemm1_variants={n1} gemm2_variants={n2} "
              f"ws {ws[0][:8]}->{ws[1][:8]}", flush=True)
        results[seed] = {"nttp": int(ntpp.item()),
                         "gemm1": variants["gemm1"],
                         "gemm2": variants["gemm2"],
                         "ws": ws}
    json.dump({"args": vars(args), "sweep": results},
              open(args.out, "w"), indent=1)
    print(f"[out] wrote {args.out}", flush=True)
    return 0


def run_matrix(args, tensors, weights, dev):
    """Bisect matrix: which knob removes multi-block slicing / changes the
    reduce path, and what does each cost in wall time (rough)."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new)
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    M, top_k, K, N = args.m, args.top_k, 2560, 640
    E = 512
    base_bs = 8
    configs = []
    for name, kw in [
        ("default", {}),
        ("fp32_reduce_off", {"no_fp32_reduce": True}),
        ("atomic_add", {"atomic_add": True}),
        ("zero_workspace", {"zero_workspace": True}),
        ("thread_128_128", {"thread_k": 128, "thread_n": 128}),
        ("thread_128_64", {"thread_k": 128, "thread_n": 64}),
        ("thread_256_64", {"thread_k": 256, "thread_n": 64}),
        ("thread_64_128_bps1", {"blocks_per_sm": 1}),
        ("bps4", {"blocks_per_sm": 4}),
        ("block16", {"force_block_size": 16}),
        ("block32", {"force_block_size": 32}),
        ("block64", {"force_block_size": 64}),
    ]:
        configs.append((name, kw))

    out = {}
    for name, kw in configs:
        bs = kw.get("force_block_size", base_bs)
        hidden, topk_ids, topk_weights = build_routing(
            tensors, dev, m=M, top_k=top_k, seed=args.seed)
        sorted_ids, expert_ids, ntpp = moe_align_block_size(
            topk_ids, bs, E, None, ignore_invalid_experts=True)
        ctx = dict(hidden=hidden, topk_ids=topk_ids,
                   topk_weights=topk_weights, sorted=sorted_ids,
                   expert_ids=expert_ids, ntpp=ntpp, block_size_m=bs,
                   m=M, top_k=top_k,
                   workspace=marlin_make_workspace_new(dev, 4))
        cache13 = torch.empty((M * top_k * max(2 * N, K),), device=dev,
                              dtype=torch.bfloat16)
        cache2f = torch.empty((M * top_k, max(N, K)), device=dev,
                              dtype=torch.bfloat16)
        ctx["cache1"] = _resize_cache(cache13, (M * top_k, 2 * N))
        ctx["cache3"] = _resize_cache(cache13, (M * top_k, K))
        ctx["cache2"] = _resize_cache(cache2f, (M * top_k, N))
        # apply the config knobs to args for this run
        saved = {k: getattr(args, k) for k in
                 ("thread_k", "thread_n", "blocks_per_sm", "no_fp32_reduce",
                  "atomic_add", "zero_workspace")}
        for k, v in kw.items():
            setattr(args, k, v)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            variants, recs, ws = tight_run(args, ctx, weights, 200)
            wall = time.perf_counter() - t0
            for k, v in saved.items():
                setattr(args, k, v)
            n1, n2 = len(variants["gemm1"]), len(variants["gemm2"])
            # rough per-call timing: 200 pairs over wall
            per_pair_ms = wall / 200 * 1e3
            print(f"[matrix] {name}: gemm1_variants={n1} "
                  f"gemm2_variants={n2} nttp={int(ntpp.item())} "
                  f"wall={wall:.2f}s per_pair_ms={per_pair_ms:.2f} "
                  f"ws {ws[0][:8]}->{ws[1][:8]}", flush=True)
            out[name] = {"gemm1": variants["gemm1"],
                         "gemm2": variants["gemm2"],
                         "nttp": int(ntpp.item()), "wall_s": wall,
                         "per_pair_ms": per_pair_ms, "ws": ws}
        except Exception as e:
            wall = time.perf_counter() - t0
            for k, v in saved.items():
                setattr(args, k, v)
            msg = str(e).splitlines()[0][:200]
            print(f"[matrix] {name}: UNSUPPORTED ({msg})", flush=True)
            out[name] = {"error": msg, "wall_s": wall}
    json.dump({"args": vars(args), "matrix": out}, open(args.out, "w"),
              indent=1)
    print(f"[out] wrote {args.out}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/model")
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hash-stage", choices=["gemm1", "gemm2", "final", "all"],
                    default="all")
    ap.add_argument("--zero-workspace", action="store_true")
    ap.add_argument("--no-fp32-reduce", action="store_true")
    ap.add_argument("--atomic-add", action="store_true")
    ap.add_argument("--thread-k", type=int, default=0)
    ap.add_argument("--thread-n", type=int, default=0)
    ap.add_argument("--blocks-per-sm", type=int, default=0)
    ap.add_argument("--force-block-size", type=int, default=0)
    ap.add_argument("--fresh-caches", action="store_true",
                    help="reallocate intermediate caches each iteration")
    ap.add_argument("--probe-ctmp", action="store_true",
                    help="after each call, re-alloc the c_tmp-sized block via "
                         "the caching allocator and hash it (separate probe "
                         "run; perturbs allocation order if combined with "
                         "the bimodality loop)")
    ap.add_argument("--tight", action="store_true",
                    help="no per-iteration host sync: launch pairs back to "
                         "back with rotating device-side output snapshots, "
                         "hash only after the final sync (mimics serving's "
                         "back-to-back kernel cadence)")
    ap.add_argument("--poison-ctmp", action="store_true",
                    help="between iterations, allocate the c_tmp-sized block "
                         "and fill it with varying garbage so the kernel's "
                         "next c_tmp allocation reuses varying stale content")
    ap.add_argument("--profile", action="store_true",
                    help="profile one iteration and dump marlin kernel names")
    ap.add_argument("--sweep-seeds", type=int, default=0,
                    help="run N seeds x 200 tight iters each (routing "
                         "histogram/geometry varies per seed), weights "
                         "loaded once")
    ap.add_argument("--matrix", action="store_true",
                    help="mechanism bisect matrix: run the tight 200-iter "
                         "block under each knob combination (single-split "
                         "forcing via thread_k/n + blocks_per_sm, "
                         "use_fp32_reduce/use_atomic_add toggles, workspace "
                         "zeroing, M padding via block size), recording "
                         "variants and rough timing per config")
    ap.add_argument("--out", default="repro_result.json")
    args = ap.parse_args()

    dev = assert_env()
    torch.manual_seed(args.seed)

    tensors = load_layer3(args.ckpt)
    weights = build_repacked_weights(tensors, dev)

    if args.sweep_seeds:
        return run_seed_sweep(args, tensors, weights, dev)

    if args.matrix:
        return run_matrix(args, tensors, weights, dev)

    ctx = build_call_ctx(args, tensors, dev)

    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types

    iters = args.iters
    assert iters >= 100, "review requires 100+ identical-input iterations"

    variants = {}
    ws_hist = {}
    times = {"gemm1": [], "gemm2": []}
    c_tmp_probe = []
    records = []

    # c_tmp size as the host wrapper computes it (ops.cu:556-565),
    # max_thread_n=256 (marlin.cuh)
    sms = ctx["sms"]
    sorted_len = ctx["sorted"].shape[0]
    bsz = ctx["block_size_m"]
    c_tmp_g1 = min(1280 * sorted_len, sms * 4 * bsz * 256)
    c_tmp_g2 = min(2560 * sorted_len, sms * 4 * bsz * 256)
    if bsz == 8:
        c_tmp_g1 *= 2
        c_tmp_g2 *= 2
    print(f"[ctx] c_tmp elems gemm1={c_tmp_g1} gemm2={c_tmp_g2} "
          f"({c_tmp_g1 * 4 / 1e6:.2f} / {c_tmp_g2 * 4 / 1e6:.2f} MB)", flush=True)

    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                run_pair(args, ctx, weights, args.hash_stage)
                torch.cuda.synchronize()
        names = sorted({e.key for e in prof.key_averages()
                        if "marlin" in e.key.lower() or "Marlin" in e.key})
        print("[profile] marlin kernels:", flush=True)
        for n in names:
            print("   ", n, flush=True)

    w13, w2, w13_scales, w2_scales = weights
    M, top_k, K, N = ctx["m"], ctx["top_k"], 2560, 640
    tk = args.thread_k or -1
    tn = args.thread_n or -1
    bps = args.blocks_per_sm or -1
    ufa = not args.no_fp32_reduce
    uaa = args.atomic_add

    def poison():
        # allocate the c_tmp-sized block, fill with iteration-varying garbage,
        # free it: the next op-internal c_tmp allocation reuses this block via
        # the caching allocator, emulating serving's varying stale leftovers
        for size in (c_tmp_g1, c_tmp_g2):
            t = torch.empty(size, dtype=torch.float32, device="cuda")
            t.fill_(float(it % 7 + 1))
            del t

    tight = args.tight
    if tight:
        snap1 = [torch.empty_like(ctx["cache1"]) for _ in range(iters)]
        snap2 = [torch.empty_like(ctx["cache3"]) for _ in range(iters)]
        ws_loop_before = sha(ctx["workspace"])

    for it in range(iters):
        if args.fresh_caches:
            reset_caches(ctx)
        if tight:
            if args.zero_workspace:
                ctx["workspace"].zero_()
            cache1 = ops.moe_wna16_marlin_gemm(
                ctx["hidden"], ctx["cache1"], w13, None, w13_scales, None,
                None, None, None, None, ctx["workspace"], ctx["sorted"],
                ctx["expert_ids"], ctx["ntpp"], ctx["topk_weights"],
                moe_block_size=ctx["block_size_m"], top_k=top_k,
                mul_topk_weights=False, b_q_type=scalar_types.uint4b8,
                size_m=M, size_n=2 * N, size_k=K, is_k_full=True,
                use_atomic_add=uaa, use_fp32_reduce=ufa, is_zp_float=False,
                thread_k=tk, thread_n=tn, blocks_per_sm=bps)
            torch.ops._C.silu_and_mul(ctx["cache2"], cache1)
            cache3 = ops.moe_wna16_marlin_gemm(
                ctx["cache2"], ctx["cache3"], w2, None, w2_scales, None,
                None, None, None, None, ctx["workspace"], ctx["sorted"],
                ctx["expert_ids"], ctx["ntpp"], ctx["topk_weights"],
                moe_block_size=ctx["block_size_m"], top_k=1,
                mul_topk_weights=True, b_q_type=scalar_types.uint4b8,
                size_m=M * top_k, size_n=K, size_k=N, is_k_full=True,
                use_atomic_add=uaa, use_fp32_reduce=ufa, is_zp_float=False,
                thread_k=tk, thread_n=tn, blocks_per_sm=bps)
            snap1[it].copy_(cache1, non_blocking=True)
            snap2[it].copy_(cache3, non_blocking=True)
            if args.poison_ctmp:
                poison()
            continue

        rec = {"i": it}
        ws_before = sha(ctx["workspace"])
        torch.cuda.synchronize()
        # time each op separately with events (no host sync between)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev2 = torch.cuda.Event(enable_timing=True)
        if args.zero_workspace:
            ctx["workspace"].zero_()
        ev0.record()
        cache1 = ops.moe_wna16_marlin_gemm(
            ctx["hidden"], ctx["cache1"], w13, None, w13_scales, None, None,
            None, None, None, ctx["workspace"], ctx["sorted"],
            ctx["expert_ids"], ctx["ntpp"], ctx["topk_weights"],
            moe_block_size=ctx["block_size_m"], top_k=top_k,
            mul_topk_weights=False, b_q_type=scalar_types.uint4b8,
            size_m=M, size_n=2 * N, size_k=K, is_k_full=True,
            use_atomic_add=uaa, use_fp32_reduce=ufa, is_zp_float=False,
            thread_k=tk, thread_n=tn, blocks_per_sm=bps)
        ev1.record()
        torch.ops._C.silu_and_mul(ctx["cache2"], cache1)
        cache3 = ops.moe_wna16_marlin_gemm(
            ctx["cache2"], ctx["cache3"], w2, None, w2_scales, None, None,
            None, None, None, ctx["workspace"], ctx["sorted"],
            ctx["expert_ids"], ctx["ntpp"], ctx["topk_weights"],
            moe_block_size=ctx["block_size_m"], top_k=1,
            mul_topk_weights=True, b_q_type=scalar_types.uint4b8,
            size_m=M * top_k, size_n=K, size_k=N, is_k_full=True,
            use_atomic_add=uaa, use_fp32_reduce=ufa, is_zp_float=False,
            thread_k=tk, thread_n=tn, blocks_per_sm=bps)
        ev2.record()
        torch.cuda.synchronize()
        times["gemm1"].append(ev0.elapsed_time(ev1))
        times["gemm2"].append(ev1.elapsed_time(ev2))

        out = ctx["output_buf"]
        torch.sum(cache3.view(-1, top_k, K), dim=1, out=out)
        hashes = {"gemm1": sha(cache1), "gemm2": sha(cache3), "final": sha(out)}
        ws_after = sha(ctx["workspace"])
        ws_key = f"{ws_before}->{ws_after}"
        ws_hist.setdefault(ws_key, 0)
        ws_hist[ws_key] += 1

        for stage, h in hashes.items():
            if args.hash_stage not in ("all", stage):
                continue
            variants.setdefault(stage, {})
            variants[stage].setdefault(h, 0)
            variants[stage][h] += 1
        rec.update(hashes)
        rec["ws"] = ws_key
        if args.probe_ctmp:
            for size in (c_tmp_g1, c_tmp_g2):
                probe = torch.empty(size, dtype=torch.float32, device="cuda")
                rec[f"ctmp{size}"] = sha(probe)
                del probe
        records.append(rec)
        if (it + 1) % 50 == 0:
            vs = {s: len(v) for s, v in variants.items()}
            print(f"[iter {it + 1}] variants={vs}", flush=True)

    if tight:
        torch.cuda.synchronize()
        ws_loop_after = sha(ctx["workspace"])
        ws_hist[f"{ws_loop_before}->{ws_loop_after} (loop)"] = iters
        for it in range(iters):
            h1, h2 = sha(snap1[it]), sha(snap2[it])
            for stage, h in (("gemm1", h1), ("gemm2", h2)):
                if args.hash_stage not in ("all", stage):
                    continue
                variants.setdefault(stage, {})
                variants[stage].setdefault(h, 0)
                variants[stage][h] += 1
            records.append({"i": it, "gemm1": h1, "gemm2": h2})
        print(f"[result] loop workspace {ws_loop_before} -> {ws_loop_after}",
              flush=True)

    summary = {}
    for stage, hist in variants.items():
        summary[stage] = {
            "variants": hist,
            "n_variants": len(hist),
            "calls": sum(hist.values()),
        }
        print(f"[result] {stage}: {len(hist)} variant(s) over "
              f"{sum(hist.values())} calls: "
              f"{sorted(hist.items(), key=lambda kv: -kv[1])}", flush=True)
    for stage in ("gemm1", "gemm2"):
        t = times[stage]
        if t:
            print(f"[result] {stage} ms: p50={statistics.median(t):.3f} "
                  f"mean={statistics.fmean(t):.3f} min={min(t):.3f}", flush=True)
    print(f"[result] workspace transitions: {ws_hist}", flush=True)

    json.dump({
        "args": vars(args),
        "env": {
            "device": torch.cuda.get_device_name(dev),
            "cc": list(torch.cuda.get_device_capability(dev)),
            "sms": ctx["sms"],
            "torch": torch.__version__,
        },
        "call_context": {
            "block_size_m": ctx["block_size_m"],
            "num_tokens_post_padded": int(ctx["ntpp"].item()),
            "sorted_sha": sha(ctx["sorted"]),
            "expert_ids_sha": sha(ctx["expert_ids"]),
            "topk_ids_sha": sha(ctx["topk_ids"]),
            "topk_weights_sha": sha(ctx["topk_weights"]),
            "hidden_sha": sha(ctx["hidden"]),
            "c_tmp_elems_g1": c_tmp_g1,
            "c_tmp_elems_g2": c_tmp_g2,
        },
        "summary": summary,
        "workspace_transitions": ws_hist,
        "timing_ms": {
            "gemm1_p50": statistics.median(times["gemm1"]) if times["gemm1"] else None,
            "gemm2_p50": statistics.median(times["gemm2"]) if times["gemm2"] else None,
        },
        "records": records if iters <= 500 else records[:500],
    }, open(args.out, "w"), indent=1)
    print(f"[out] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
