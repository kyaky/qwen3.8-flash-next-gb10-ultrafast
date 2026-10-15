#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Append the MARLIN_SHIM wrapper to vllm/_custom_ops.py inside a throwaway
container. Modes (env MARLIN_SHIM, comma-separated):
  sortedlog  log sha256 of sorted_token_ids/expert_ids per call (raw align output)
  zerows     zero the workspace (locks) before every call
  stablesort deterministically re-sort each expert group's valid entries
             ascending in sorted_token_ids before the call (size_m<=512 only)
  atomic     force use_atomic_add=True
  fp32off    force use_fp32_reduce=False
  wslog      hash the workspace before/after each call to MARLIN_SHIM_LOG
  dump       save any output whose hash differs from the first-seen hash for
             (shape, input-hash) as a .pt
  dumpin     save the first input tensor per (shape, input-hash)
  tk128_128 / tk128_64 / bps1 / bps4  force the launch-selection knobs
"""
import os

SP = "/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py"
SHIM = '''

# --- marlin repro bisect shim (throwaway container only; D-011) ---
import os as _ms_os
_ms_mode = _ms_os.environ.get("MARLIN_SHIM", "")
if _ms_mode:
    import hashlib as _ms_hl
    import json as _ms_json
    import time as _ms_time
    _ms_log = open(_ms_os.environ.get("MARLIN_SHIM_LOG",
                                      "/detout/shim.jsonl"), "a", buffering=1)
    _ms_ci = [0]
    _ms_dump = {}
    _ms_dumpin = set()

    def moe_wna16_marlin_gemm(
        input, output, b_qweight, b_bias, b_scales, a_scales, global_scale,
        b_qzeros, g_idx, perm, workspace, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, b_q_type, size_m, size_n, size_k, is_k_full,
        use_atomic_add, use_fp32_reduce, is_zp_float, thread_k=-1,
        thread_n=-1, blocks_per_sm=-1,
    ):
        _small = size_m <= 512
        if "sortedlog" in _ms_mode and _small:
            _ssh = _ms_hl.sha256(
                sorted_token_ids.detach().cpu().view(torch.uint8)
                .numpy().tobytes()).hexdigest()[:12]
            _esh = _ms_hl.sha256(
                expert_ids.detach().cpu().view(torch.uint8)
                .numpy().tobytes()).hexdigest()[:12]
            _ms_log.write(_ms_json.dumps({
                "t": round(_ms_time.time(), 4), "ci": _ms_ci[0],
                "sm": size_m, "ntpp": int(num_tokens_past_padded.item()),
                "sorted_sha": _ssh, "expert_sha": _esh}) + "\\n")
        if "zerows" in _ms_mode:
            workspace.zero_()
        if "stablesort" in _ms_mode and _small:
            _SENT = size_m * top_k
            _ei = expert_ids.detach().cpu().tolist()
            _bsz = int(moe_block_size)
            _nb = sorted_token_ids.shape[0] // _bsz
            _host = sorted_token_ids.detach().cpu()
            _out = _host.clone()
            _i = 0
            while _i < _nb:
                _j = _i
                while _j + 1 < _nb and _ei[_j + 1] == _ei[_i]:
                    _j += 1
                _seg = _host[_i * _bsz:(_j + 1) * _bsz]
                _v = int((_seg < _SENT).sum().item())
                if _v > 1:
                    _out[_i * _bsz:_i * _bsz + _v] = _seg[:_v].sort().values
                _i = _j + 1
            sorted_token_ids.copy_(_out.to(sorted_token_ids.device),
                                   non_blocking=True)
        ua = True if "atomic" in _ms_mode else use_atomic_add
        ur = False if "fp32off" in _ms_mode else use_fp32_reduce
        tk, tn, bps = thread_k, thread_n, blocks_per_sm
        if "tk128_128" in _ms_mode:
            tk, tn = 128, 128
        if "tk128_64" in _ms_mode:
            tk, tn = 128, 64
        if "bps1" in _ms_mode:
            bps = 1
        if "bps4" in _ms_mode:
            bps = 4
        _need_ih = ("dump" in _ms_mode or "dumpin" in _ms_mode
                    or "wslog" in _ms_mode)
        if _need_ih and _small:
            _ih = _ms_hl.sha256(
                input.detach().cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest()[:12]
            _shp = (size_m, size_n, size_k, top_k)
        if "wslog" in _ms_mode and _small:
            _w0 = workspace.detach().cpu().view(torch.uint8).numpy().tobytes()
            _t0 = _ms_time.time()
        out = torch.ops._moe_C.moe_wna16_marlin_gemm(
            input, output, b_qweight, b_bias, b_scales, a_scales, global_scale,
            b_qzeros, g_idx, perm, workspace, sorted_token_ids, expert_ids,
            num_tokens_past_padded, topk_weights, moe_block_size, top_k,
            mul_topk_weights, b_q_type.id, size_m, size_n, size_k, is_k_full,
            ua, ur, is_zp_float, tk, tn, bps)
        if "dump" in _ms_mode and _small:
            _h = _ms_hl.sha256(
                out.detach().cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest()[:16]
            _key = (_shp, _ih)
            _prev = _ms_dump.setdefault(_key, _h)
            if _h != _prev:
                _p = "%s/dump_s%dx%dx%d_t%d_in%s_%s.pt" % (
                    _ms_os.environ.get("MARLIN_SHIM_LOG_DIR", "/detout"),
                    _shp[0], _shp[1], _shp[2], _shp[3], _ih, _h)
                torch.save(out.detach().cpu(), _p)
                _ms_log.write(_ms_json.dumps({
                    "t": round(_ms_time.time(), 4), "ci": _ms_ci[0],
                    "sm": size_m, "ih": _ih, "event": "variant",
                    "hash": _h, "first": _prev, "saved": _p}) + "\\n")
        if "dumpin" in _ms_mode and _small and _key not in _ms_dumpin:
            _dir = _ms_os.environ.get("MARLIN_SHIM_LOG_DIR", "/detout")
            _p = "%s/in_s%dx%dx%d_t%d_in%s.pt" % (
                _dir, _shp[0], _shp[1], _shp[2], _shp[3], _ih)
            torch.save({"input": input.detach().cpu(),
                        "sorted": sorted_token_ids.detach().cpu(),
                        "expert_ids": expert_ids.detach().cpu(),
                        "ntpp": num_tokens_past_padded.detach().cpu(),
                        "topk_weights": topk_weights.detach().cpu(),
                        "mul_topk_weights": mul_topk_weights,
                        "moe_block_size": moe_block_size,
                        "b_qweight_shape": list(b_qweight.shape),
                        "b_scales_shape": list(b_scales.shape),
                        "b_scales_dtype": str(b_scales.dtype),
                        "workspace_hash": _ms_hl.sha256(
                            workspace.detach().cpu().view(torch.uint8)
                            .numpy().tobytes()).hexdigest()[:16]}, _p)
            _ms_dumpin.add(_key)
        if "wslog" in _ms_mode and _small:
            _w1 = workspace.detach().cpu().view(torch.uint8).numpy().tobytes()
            _ms_log.write(_ms_json.dumps({
                "t": round(_ms_time.time(), 4), "ci": _ms_ci[0],
                "sm": size_m, "ih": _ih,
                "ntpp": int(num_tokens_past_padded.item()),
                "ws0": _ms_hl.sha256(_w0).hexdigest()[:16],
                "ws1": _ms_hl.sha256(_w1).hexdigest()[:16],
                "ws_nz": int((workspace != 0).sum().item()),
                "mode": _ms_mode}) + "\\n")
        _ms_ci[0] += 1
        return out
'''
with open(SP, "a") as f:
    f.write(SHIM)
import ast
ast.parse(open(SP).read())
print("shim appended:", os.environ.get("MARLIN_SHIM", ""))
