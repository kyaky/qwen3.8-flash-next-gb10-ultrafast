#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Faster MoE weight loading on GB10 (VLLM_LOAD_CLONE=1; 0 disables).

Retargeted to the dev snapshot, where ``expert_data.copy_(loaded_weight)`` appears at FIVE
sites in routed_experts.py (the v0.30 layout had two: the final copies of _load_w13 /
_load_w2). The extra sites live in the per-shard / fused branches. We wrap ALL FIVE
with the same helper: it is an identity for any source that is not a CPU unpinned
(file-backed mmap) tensor, so semantics are unchanged and whichever sites the
fastsafetensors loader exercises are all covered. The clone gives byte-identical
weights (one transient tensor copy) and avoids the ~1.7 ms/tensor pageable-H2D slow
path (~0.23 ms from anonymous memory).
Usage: python3 patch_moe_load_clone.py <site-packages-dir>
"""
import ast
import sys

RE = f"{sys.argv[1]}/vllm/model_executor/layers/fused_moe/routed_experts.py"
s = open(RE).read()

OLD = "expert_data.copy_(loaded_weight)"
n = s.count(OLD)
assert n == 5, f"expected 5 expert_data.copy_(loaded_weight) sites in {RE}, found {n}"
s = s.replace(OLD, "expert_data.copy_(_qwen38_h2d_src(expert_data, loaded_weight))")

s += '''

# --- gb10-flashnext: clone mmap-backed weights before the H2D copy (VLLM_LOAD_CLONE=1) ---
import os as _qwen38_os

_QWEN38_LOAD_CLONE = _qwen38_os.environ.get("VLLM_LOAD_CLONE", "1") != "0"


def _qwen38_h2d_src(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if (
        _QWEN38_LOAD_CLONE
        and dst.device.type != "cpu"
        and src.device.type == "cpu"
        and not src.is_pinned()
    ):
        return src.clone()
    return src
'''

ast.parse(s)
open(RE, "w").write(s)
print(f"routed_experts.py: MoE load clone applied OK ({n} sites wrapped)")
