#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Env-gated switch of the QSA block top-k to the deterministic kernel (vllm#55122),
retargeted to the dev-snapshot layout, where the dispatch lives in
``ops/qsa.py::qsa_select_paged_tokens`` (shared prefill+decode call site via nvidia/indexer_qsa.py).
  VLLM_QSA_DET_TOPK=1     use torch.ops._C_det.persistent_topk instead of _C.persistent_topk
  VLLM_QSA_DET_LIB=<path> the .so (default /opt/llm/kernel-det/_C_det.so)
Inert without VLLM_QSA_DET_TOPK. A re-run without "off" fails loudly (anchor no longer present);
run with "off" to restore the stock dispatch.
Usage: python3 patch_qsadet.py <ops/qsa.py> [off]
"""
import sys

TARGET = sys.argv[1]
# (indent, function that owns the dispatch and carries the _qsadet_loaded flag)
LAYOUTS = [("        ", "qsa_select_paged_tokens")]


def _blocks(indent, holder):
    anchor = (
        f"{indent}topk_op = (\n"
        f"{indent}    torch.ops._C.cooperative_topk\n"
        f"{indent}    if use_cooperative_topk\n"
        f"{indent}    else torch.ops._C.persistent_topk\n"
        f"{indent})\n"
    )
    new = (
        f"{indent}# ---- QSADET (deterministic top-k wiring, 2026-10-12) ----\n"
        f"{indent}import os as _os\n"
        f"{indent}if _os.environ.get(\"VLLM_QSA_DET_TOPK\"):\n"
        f"{indent}    if not getattr({holder}, \"_qsadet_loaded\", False):\n"
        f"{indent}        _lib = _os.environ.get(\"VLLM_QSA_DET_LIB\", \"/opt/llm/kernel-det/_C_det.so\")\n"
        f"{indent}        torch.ops.load_library(_lib)\n"
        f"{indent}        {holder}._qsadet_loaded = True\n"
        f"{indent}        print(f\"QSADET active: {{_lib}}\", flush=True)\n"
        f"{indent}    topk_op = torch.ops._C_det.persistent_topk\n"
        f"{indent}else:\n"
        f"{indent}    topk_op = (\n"
        f"{indent}        torch.ops._C.cooperative_topk\n"
        f"{indent}        if use_cooperative_topk\n"
        f"{indent}        else torch.ops._C.persistent_topk\n"
        f"{indent}    )\n"
        f"{indent}# ---- end QSADET ----\n"
    )
    return anchor, new


s = open(TARGET).read()
off = sys.argv[2:] and sys.argv[2] == "off"
for indent, holder in LAYOUTS:
    anchor, new = _blocks(indent, holder)
    if off and new in s:
        open(TARGET, "w").write(s.replace(new, anchor)); print("  qsadet REMOVED"); break
    if not off and s.count(anchor) == 1 and f"def {holder}(" in s:
        if "QSADET" in s:
            print("  qsadet already installed"); break
        open(TARGET, "w").write(s.replace(anchor, new))
        print(f"  qsadet INSTALLED in {TARGET} (holder {holder}; inert unless VLLM_QSA_DET_TOPK=1)"); break
else:
    raise SystemExit("qsadet: anchor not found in " + TARGET)
