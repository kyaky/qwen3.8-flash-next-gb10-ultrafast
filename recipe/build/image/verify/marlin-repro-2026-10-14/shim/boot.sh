#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Throwaway replica boot with optional marlin bisect shim. Mirrors the
# env-drill3 serving flags (minus speculative decode, plus DET probes).
set -e
SP=/usr/local/lib/python3.12/dist-packages
cp "$SP/vllm/_custom_ops.py" "$SP/vllm/_custom_ops.py.pristine"
if [ -n "${MARLIN_SHIM:-}" ]; then
  python3 -I /bisect/append_shim.py
fi
if [ -n "${MARLIN_FORCE_BLOCK:-}" ]; then
  python3 - "$SP/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py" <<'PYEOF'
import re, sys, os
p = sys.argv[1]
s = open(p).read()
old = """    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break"""
new = """    _fb = int(__import__("os").environ.get("MARLIN_FORCE_BLOCK", "0"))
    if _fb:
        block_size_m = _fb
    else:
        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break"""
assert old in s, "block-size loop not found"
open(p, "w").write(s.replace(old, new))
import ast; ast.parse(open(p).read())
print("marlin_moe block-size patched:", os.environ.get("MARLIN_FORCE_BLOCK"))
PYEOF
fi
exec vllm serve /model --served-model-name qwen --host 0.0.0.0 --port 8000 \
  --load-format fastsafetensors --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.01 --enable-prefix-caching \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --kv-cache-dtype auto --kv-cache-memory-bytes 1g \
  --enable-prompt-tokens-details --enforce-eager
