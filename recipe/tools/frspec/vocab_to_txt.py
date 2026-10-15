#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a build_token_map.py torch.save output into the draft-vocab
text format (one integer id per line, consumed gzipped as VLLM_MTP_DRAFT_VOCAB;
launch.sh validates exactly `size` lines). Verifies id range against the model
vocab_size before writing.

Usage: vocab_to_txt.py <map.pt> <out.txt.gz> <vocab_size>
"""
import gzip
import sys
import torch

src, dst, vocab_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = torch.load(src, weights_only=True)
assert isinstance(ids, list) and len(ids) == len(set(ids)), "malformed map"
assert all(isinstance(i, int) and 0 <= i < vocab_size for i in ids), "id out of range"
payload = "".join(f"{i}\n" for i in sorted(ids))
with gzip.open(dst, "wt", compresslevel=9) as fh:
    fh.write(payload)
with gzip.open(dst, "rt") as fh:
    lines = fh.read().splitlines()
assert len(lines) == len(ids) and [int(x) for x in lines] == sorted(ids)
print(f"vocab_to_txt OK: {len(ids)} ids -> {dst}")
