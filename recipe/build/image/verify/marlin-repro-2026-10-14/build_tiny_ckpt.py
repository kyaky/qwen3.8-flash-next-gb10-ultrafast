#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a 4-layer qwen4_exp mini checkpoint from the real W4A16 checkpoint.

Keeps: all non-layer tensors (embeddings, lm_head int8-gptq, hyper-connection
mixer, vision tower, MTP draft), plus layers 0..3 verbatim (layer 3 is the
D-011 target: full-attention + MoE marlin experts). Quantization config and
tokenizer files are copied unchanged; only num_hidden_layers / layer_types
are truncated so the serving stack, loader and marlin call context are the
real ones.
"""
import json
import os
import shutil
import struct
import sys

import torch
from safetensors.torch import save_file

SRC = "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32"
DST = sys.argv[1] if len(sys.argv) > 1 else "tiny-ckpt"
LAYERS = 4
SHARD_BYTES = 2 * 2**30


def main():
    os.makedirs(DST, exist_ok=True)
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    keep = {}
    for k, f in idx.items():
        if ".layers." not in k:
            keep[k] = f
        else:
            L = int(k.split(".layers.")[1].split(".")[0])
            if L < LAYERS:
                keep[k] = f
    assert len(keep) > 20000, len(keep)

    cfg = json.load(open(f"{SRC}/config.json"))
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = LAYERS
    tc["layer_types"] = tc["layer_types"][:LAYERS]
    ple_ids = tc.get("ple_layer_ids")
    if ple_ids:
        tc["ple_layer_ids"] = [i for i in ple_ids if i < LAYERS]
    json.dump(cfg, open(f"{DST}/config.json", "w"), indent=1)

    # stream-copy the selected tensors into fresh shards
    need = {}
    for k, f in keep.items():
        need.setdefault(f, []).append(k)
    index = {}
    shard, shard_size, tensors, shard_idx = 0, 0, {}, 0

    def flush():
        nonlocal shard, shard_size, tensors, shard_idx
        if tensors:
            name = f"tiny-{shard_idx:05d}-of-XXXXX.safetensors"
            path = f"{DST}/{name}"
            save_file(tensors, path, metadata={"format": "pt"})
            for k in tensors:
                index[k] = name
            print(f"  wrote {name} ({shard_size / 2**30:.2f} GiB, "
                  f"{len(tensors)} tensors)", flush=True)
        shard, shard_size, tensors, shard_idx = shard_idx + 1, 0, {}, shard_idx + 1

    for f, keys in sorted(need.items()):
        with open(f"{SRC}/{f}", "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
            body_off = 8 + n
            spans = []
            for k in keys:
                a, b = header[k]["data_offsets"]
                spans.append((a, b, k))
            spans.sort()
            pos = 0
            for a, b, k in spans:
                if b - a > SHARD_BYTES:
                    raise RuntimeError(f"{k} too big")
                if shard_size + (b - a) > SHARD_BYTES and tensors:
                    flush()
                fh.seek(body_off + a)
                blob = fh.read(b - a)
                dtype = header[k]["dtype"]
                dt = {"F64": torch.float64, "F32": torch.float32,
                      "F16": torch.float16, "BF16": torch.bfloat16,
                      "I64": torch.int64, "I32": torch.int32,
                      "I16": torch.int16, "I8": torch.int8,
                      "U8": torch.uint8, "BOOL": torch.bool,
                      "F8_E4M3": torch.float8_e4m3fn,
                      "F8_E5M2": torch.float8_e5m2}[dtype]
                t = torch.frombuffer(bytearray(blob), dtype=dt)
                t = t.reshape(header[k]["shape"]).clone()
                tensors[k] = t
                shard_size += b - a
                pos = b
    flush()

    # fix shard count in names
    total = shard
    renames = {}
    for k, name in index.items():
        renames.setdefault(name, name.replace("of-XXXXX", f"of-{total:05d}"))
    for old, new in renames.items():
        if old != new:
            os.rename(f"{DST}/{old}", f"{DST}/{new}")
    for k in index:
        index[k] = index[k].replace("of-XXXXX", f"of-{total:05d}")
    json.dump({"metadata": {"total_size": None}, "weight_map": index},
              open(f"{DST}/model.safetensors.index.json", "w"))

    for f in os.listdir(SRC):
        if f.endswith((".json", ".jinja", ".txt", ".model")) and \
                not f.startswith("model-") and \
                f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
            print("  copied", f)
    print(f"[done] {len(index)} tensors -> {DST}")


if __name__ == "__main__":
    main()
