#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU gates for the iter6e image (deterministic QSA top-k stage). Run inside the
image with the repo's recipe/build/image/src mounted read-only at /selftest/src.
Checks:
  1. qsa.py carries exactly one QSADET dispatch, stock else-branch intact, .orig backup
  2. off -> install cycle restores the exact stock bytes (idempotency of the cycle)
  3. torch.ops._C_det.persistent_topk resolves from /opt/llm/kernel-det/_C_det.so
     (dlopen works without a GPU; CUDA-only impl is exercised at boot by smoke-test)
  4. the loaded op schema matches the 6-arg call site
"""
import ast
import os
import subprocess
import sys

SP = "/usr/local/lib/python3.12/dist-packages"
QSA = f"{SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
RE = f"{SP}/vllm/model_executor/layers/fused_moe/routed_experts.py"
VPE = f"{SP}/vllm/model_executor/layers/vocab_parallel_embedding.py"
PATCH = "/selftest/src/patch_qsadet.py"
SO = "/opt/llm/kernel-det/_C_det.so"
ANCHOR = (
    "        topk_op = (\n"
    "            torch.ops._C.cooperative_topk\n"
    "            if use_cooperative_topk\n"
    "            else torch.ops._C.persistent_topk\n"
    "        )\n"
)
ELSE_BRANCH = (
    "        else:\n"
    "            topk_op = (\n"
    "                torch.ops._C.cooperative_topk\n"
    "                if use_cooperative_topk\n"
    "                else torch.ops._C.persistent_topk\n"
    "            )\n"
)

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


s = open(QSA).read()
check(s.count("QSADET") >= 2 and s.count("# ---- QSADET (") == 1, "exactly one QSADET dispatch block")
check(s.count(ELSE_BRANCH) == 1 and ANCHOR not in s,
      "stock topk_op dispatch preserved exactly once inside the QSADET else-branch")
check(os.path.exists(QSA + ".orig"), "qsa.py.orig backup present")
ast.parse(s)
check(True, "patched qsa.py parses")
orig = open(QSA + ".orig").read()
check(orig.count(ANCHOR) == 1 and "QSADET" not in orig, "backup is the clean stock file")

# cycle: off -> install must reproduce the patched bytes exactly
subprocess.run([sys.executable, PATCH, QSA, "off"], check=True, capture_output=True)
check(open(QSA).read() == orig, "off restores byte-identical stock qsa.py")
subprocess.run([sys.executable, PATCH, QSA], check=True, capture_output=True)
check(open(QSA).read() == s, "re-install reproduces the patched bytes exactly")

import torch  # noqa: E402

torch.ops.load_library(SO)
op = torch.ops._C_det.persistent_topk.default
schema = str(op._schema)
print("schema:", schema)
check("persistent_topk" in schema, "_C_det.persistent_topk resolves from the vendored .so")
check(
    [a.name for a in op._schema.arguments] ==
    ["logits", "lengths", "output", "workspace", "k", "max_seq_len"],
    "schema argument names match the 6-arg call site",
)

# loader subset gates
re_s = open(RE).read()
check(re_s.count("_qwen38_h2d_src(expert_data, loaded_weight)") == 5,
      "all 5 expert copy sites wrapped with the h2d helper")
check("def _qwen38_h2d_src" in re_s, "h2d helper defined")
check("qwen38-moe-name-index-begin" in re_s and "_qwen38_expert_index" in re_s,
      "name-index block present")
check("expert_data.copy_(loaded_weight)" not in re_s, "no unwrapped expert copy sites remain")
vpe_s = open(VPE).read()
check("def _qwen38_chunked_copy" in vpe_s, "embed chunked-copy helper defined")
ast.parse(re_s); ast.parse(vpe_s)
check(True, "loader-patched files parse")

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
