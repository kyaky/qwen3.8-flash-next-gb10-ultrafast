#!/usr/bin/env python3
"""Copy the vocabulary embeddings to the GPU in 64 MiB pieces (VLLM_LOAD_EMBED_CHUNK=1, default on).

VocabParallelEmbedding.weight_loader (embed_tokens and ParallelLMHead / lm_head) copies the whole
1.2 GiB checkpoint tensor to the GPU with one copy_ straight from the safetensors mmap view: the
same file-backed H2D slow path as patch 14, ~150 MB/s, ~8 s per tensor, paid four times per boot
(target + MTP drafter). Each 64 MiB piece is cloned into ordinary memory first; peak extra memory is
one piece. Sources already in ordinary memory (patch 15's pread tensors) are copied directly. The
PLE table is not a VocabParallelEmbedding (vllm_ple_mmap swaps in a placeholder), so it never gets
here."""
import ast
import sys

VPE = f"{sys.argv[1]}/vllm/model_executor/layers/vocab_parallel_embedding.py"
s = open(VPE).read()
old = "        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)\n"
assert s.count(old) == 1, f"VocabParallelEmbedding.weight_loader copy: expected 1, found {s.count(old)}"
s = s.replace(old, "        _qwen38_chunked_copy(param[: loaded_weight.shape[0]].data, loaded_weight)\n")
s += '''

# --- qwen38-flash-dgx: copy large CPU weights to the GPU in 64 MiB pieces (VLLM_LOAD_EMBED_CHUNK=1) ---
# qwen38-embed-chunk-begin
import os as _qwen38_os

_QWEN38_EMBED_CHUNK = _qwen38_os.environ.get("VLLM_LOAD_EMBED_CHUNK", "1") != "0"
_QWEN38_CHUNK_BYTES = 64 << 20


def _qwen38_copy_in_chunks(dst: torch.Tensor, src: torch.Tensor) -> None:
    row = max(1, src[0].numel() * src.element_size())
    step = max(1, _QWEN38_CHUNK_BYTES // row)
    for i in range(0, src.shape[0], step):
        dst[i : i + step].copy_(src[i : i + step].clone())


def _qwen38_chunked_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    if (
        not _QWEN38_EMBED_CHUNK
        or dst.device.type == "cpu"
        or src.device.type != "cpu"
        or src.dim() == 0
        or dst.shape != src.shape
        or src.is_pinned()
        or getattr(src.untyped_storage(), "_qwen38_anon", False)
    ):
        dst.copy_(src)
        return
    _qwen38_copy_in_chunks(dst, src)
# qwen38-embed-chunk-end
'''
ast.parse(s)
open(VPE, "w").write(s)
print("vocab_parallel_embedding.py: chunked embedding copy applied OK")
