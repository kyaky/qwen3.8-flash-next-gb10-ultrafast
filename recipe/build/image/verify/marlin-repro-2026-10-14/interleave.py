#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D-011 follow-up probe: multi-layer marlin MoE interleave.

Emulates the serving forward cadence (96 moe_wna16_marlin_gemm calls per
forward across 48 layers, each with its own workspace + provisioned
intermediate caches, c_tmp freshly allocated per call from the shared
caching allocator) with real checkpoint weights for N layers, plus an
optional big-M (chunked-prefill-shaped, moe_block_size=64) pass injected
every K iterations like the load-time dummy/first-request forwards.

No per-iteration host sync: per-layer gemm1 outputs go to rotating
device-side snapshots; a final sync hashes everything. Workspace (locks)
state is checked at chosen intervals.
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


def load_layers(ckpt, layers):
    from safetensors import safe_open
    import json as _json
    idx = _json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
    want = {}
    for L in layers:
        for k, f in idx.items():
            if f".layers.{L}.mlp" in k and ("experts." in k
                                            or k.endswith("mlp.gate.weight")):
                want[k] = f
    per_file = {}
    for k, f in want.items():
        per_file.setdefault(f, []).append(k)
    tensors = {L: {} for L in layers}
    for f, keys in per_file.items():
        with safe_open(f"{ckpt}/{f}", framework="pt") as fh:
            for k in keys:
                L = int(k.split(".layers.")[1].split(".")[0])
                tensors[L][k] = fh.get_tensor(k)
    for L in layers:
        n_exp = len({k.split(".experts.")[1].split(".")[0]
                     for k in tensors[L] if ".experts." in k})
        assert n_exp == 512, (L, n_exp)
    print(f"[load] layers {layers}: "
          f"{[len(tensors[L]) for L in layers]} tensors", flush=True)
    return tensors


def repack_layer(ts, dev, group_size=128):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    E, K, N = 512, 2560, 640
    pf = 8
    pfx = None
    for k in ts:
        if ".experts.0.gate_proj.qweight" in k:
            pfx = k.split(".experts.0.")[0]
            break
    assert pfx

    def stack(name):
        return torch.stack([ts[f"{pfx}.experts.{e}.{name}"]
                            for e in range(E)]).to(dev)

    gate_qw, up_qw, down_qw = (stack("gate_proj.qweight"),
                               stack("up_proj.qweight"),
                               stack("down_proj.qweight"))
    gate_sc, up_sc, down_sc = (stack("gate_proj.scales"),
                               stack("up_proj.scales"),
                               stack("down_proj.scales"))
    qz = stack("gate_proj.qzeros")[:8]
    assert bool((qz == qz.flatten()[0]).all()), "qzeros not constant"
    w13_qw = torch.cat([gate_qw, up_qw], dim=2).contiguous()
    w13_sc = torch.cat([gate_sc, up_sc], dim=2).contiguous().to(torch.bfloat16)
    down_sc = down_sc.to(torch.bfloat16)
    assert marlin_utils.marlin_moe_padded_intermediate(N, group_size) == N
    perm = torch.empty((E, 0), dtype=torch.int32, device=dev)
    w13 = ops.gptq_marlin_moe_repack(w13_qw, perm, K, 2 * N, 4, False)
    w2 = ops.gptq_marlin_moe_repack(down_qw, perm, N, K, 4, False)
    w13_s = marlin_utils.marlin_moe_permute_scales(
        s=w13_sc, size_k=N, size_n=w13_sc.shape[2], group_size=group_size,
        is_a_8bit=False)
    w2_s = marlin_utils.marlin_moe_permute_scales(
        s=down_sc, size_k=down_sc.shape[1] * group_size,
        size_n=down_sc.shape[2], group_size=group_size, is_a_8bit=False)
    gate = ts[f"{pfx}.gate.weight"].to(dev)
    return (w13, w2, w13_s, w2_s, gate)


def routing(gate, m, top_k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn((m, 2560), generator=g, dtype=torch.float32,
                         device="cpu").to(torch.bfloat16).to("cuda")
    scores = torch.softmax(hidden @ gate.t(), dim=-1)
    tw, ti = torch.topk(scores, top_k, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    return hidden, ti.to(torch.int32), tw.to(torch.float32).contiguous()


def build_layer_ctx(w, dev, m, top_k, seed, block_size_m):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new)
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    w13, w2, w13_s, w2_s, gate = w
    K, N = 2560, 640
    hidden, topk_ids, topk_weights = routing(gate, m, top_k, seed)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(
        topk_ids, block_size_m, 512, None, ignore_invalid_experts=True)
    cache13 = torch.empty((m * top_k * max(2 * N, K),), device=dev,
                          dtype=torch.bfloat16)
    cache2f = torch.empty((m * top_k, max(N, K)), device=dev,
                          dtype=torch.bfloat16)
    return dict(
        w13=w13, w2=w2, w13_s=w13_s, w2_s=w2_s,
        hidden=hidden, topk_ids=topk_ids, topk_weights=topk_weights,
        sorted=sorted_ids, expert_ids=expert_ids, ntpp=ntpp,
        block_size_m=block_size_m, m=m, top_k=top_k,
        workspace=marlin_make_workspace_new(dev, 4),
        cache1=_resize_cache(cache13, (m * top_k, 2 * N)),
        cache3=_resize_cache(cache13, (m * top_k, K)),
        cache2=_resize_cache(cache2f, (m * top_k, N)),
        output=torch.empty((m, K), device=dev, dtype=torch.bfloat16),
    )


def gemm_pair(ctx, tk, tn, bps, ufa, uaa):
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types
    M, top_k, K, N = ctx["m"], ctx["top_k"], 2560, 640
    c1 = ops.moe_wna16_marlin_gemm(
        ctx["hidden"], ctx["cache1"], ctx["w13"], None, ctx["w13_s"], None,
        None, None, None, None, ctx["workspace"], ctx["sorted"],
        ctx["expert_ids"], ctx["ntpp"], ctx["topk_weights"],
        moe_block_size=ctx["block_size_m"], top_k=top_k,
        mul_topk_weights=False, b_q_type=scalar_types.uint4b8,
        size_m=M, size_n=2 * N, size_k=K, is_k_full=True,
        use_atomic_add=uaa, use_fp32_reduce=ufa, is_zp_float=False,
        thread_k=tk, thread_n=tn, blocks_per_sm=bps)
    torch.ops._C.silu_and_mul(ctx["cache2"], c1)
    c3 = ops.moe_wna16_marlin_gemm(
        ctx["cache2"], ctx["cache3"], ctx["w2"], None, ctx["w2_s"], None,
        None, None, None, None, ctx["workspace"], ctx["sorted"],
        ctx["expert_ids"], ctx["ntpp"], ctx["topk_weights"],
        moe_block_size=ctx["block_size_m"], top_k=1,
        mul_topk_weights=True, b_q_type=scalar_types.uint4b8,
        size_m=M * top_k, size_n=K, size_k=N, is_k_full=True,
        use_atomic_add=uaa, use_fp32_reduce=ufa, is_zp_float=False,
        thread_k=tk, thread_n=tn, blocks_per_sm=bps)
    torch.sum(c3.view(-1, top_k, K), dim=1, out=ctx["output"])
    return c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/model")
    ap.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap-P", type=int, default=128)
    ap.add_argument("--big-every", type=int, default=0,
                    help="inject a big-M forward every K iters (0=never)")
    ap.add_argument("--big-m", type=int, default=2048)
    ap.add_argument("--ws-check-every", type=int, default=25)
    ap.add_argument("--zero-workspace", action="store_true")
    ap.add_argument("--no-fp32-reduce", action="store_true")
    ap.add_argument("--atomic-add", action="store_true")
    ap.add_argument("--thread-k", type=int, default=0)
    ap.add_argument("--thread-n", type=int, default=0)
    ap.add_argument("--blocks-per-sm", type=int, default=0)
    ap.add_argument("--force-block-size", type=int, default=0)
    ap.add_argument("--out", default="interleave_result.json")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    print(f"[env] {torch.cuda.get_device_name(dev)} sms="
          f"{torch.cuda.get_device_properties(dev).multi_processor_count} "
          f"free={free / 2**30:.1f}GiB total={total / 2**30:.1f}GiB",
          flush=True)
    layers = [int(x) for x in args.layers.split(",")]
    tensors = load_layers(args.ckpt, layers)
    weights = {L: repack_layer(tensors[L], dev) for L in layers}
    del tensors

    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    tk, tn = args.thread_k or -1, args.thread_n or -1
    bps = args.blocks_per_sm or -1
    ufa, uaa = not args.no_fp32_reduce, args.atomic_add

    small_bs = args.force_block_size or 8
    ctxs = {L: build_layer_ctx(weights[L], dev, args.m, args.top_k,
                               args.seed + i, small_bs)
            for i, L in enumerate(layers)}
    big_ctxs = None
    if args.big_every:
        big_bs = 64
        big_ctxs = {L: build_layer_ctx(weights[L], dev, args.big_m,
                                       args.top_k, args.seed + 100 + i,
                                       big_bs) for i, L in enumerate(layers)}

    P = min(args.snap_P, args.iters)
    snaps = {L: [torch.empty_like(ctxs[L]["cache1"]) for _ in range(P)]
             for L in layers}
    ws_log = []
    variants = {L: {} for L in layers}

    t0 = time.perf_counter()
    for it in range(args.iters):
        if big_ctxs and args.big_every and it % args.big_every == 0:
            for L in layers:
                gemm_pair(big_ctxs[L], tk, tn, bps, ufa, uaa)
        for L in layers:
            if args.zero_workspace:
                ctxs[L]["workspace"].zero_()
            c1 = gemm_pair(ctxs[L], tk, tn, bps, ufa, uaa)
            snaps[L][it % P].copy_(c1, non_blocking=True)
        if args.ws_check_every and (it + 1) % args.ws_check_every == 0:
            torch.cuda.synchronize()
            for L in layers:
                nz = int((ctxs[L]["workspace"] != 0).sum().item())
                if nz:
                    ws_log.append({"it": it, "layer": L, "nonzero": nz,
                                   "ws": sha(ctxs[L]["workspace"])})
                    print(f"[ws!] it={it} layer={L} nonzero_locks={nz}",
                          flush=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    for L in layers:
        for i in range(P):
            h = sha(snaps[L][i])
            variants[L].setdefault(h, 0)
            variants[L][h] += 1
    summary = {}
    for L in layers:
        hist = variants[L]
        summary[L] = hist
        flag = "  <== BIMODAL" if len(hist) > 1 else ""
        print(f"[result] layer {L}: {len(hist)} variant(s) "
              f"({sorted(hist.items(), key=lambda kv: -kv[1])[:4]}){flag}",
              flush=True)
    print(f"[result] wall={wall:.1f}s iters={args.iters} "
          f"pairs={len(layers) * args.iters}", flush=True)
    print(f"[result] workspace anomalies: {len(ws_log)}", flush=True)

    json.dump({"args": vars(args), "summary": {str(k): v for k, v in summary.items()},
               "ws_log": ws_log,
               "env": {"sms": sms, "torch": torch.__version__}},
              open(args.out, "w"), indent=1)
    print(f"[out] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
