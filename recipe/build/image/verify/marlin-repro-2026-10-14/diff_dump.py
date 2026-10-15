#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diff the dumped marlin output variants: how many elements differ, where,
and by how much. Usage: diff_dump.py <dump_dir>
"""
import glob
import json
import sys

import torch


def main():
    d = sys.argv[1]
    groups = {}
    for p in sorted(glob.glob(f"{d}/dump_ci*_m*_*.pt")):
        name = p.split("/")[-1]
        ci = int(name.split("_")[1][2:])
        m = int(name.split("_m")[1].split("_")[0])
        h = name.split("_")[-1].split(".")[0]
        groups.setdefault((ci, m), {})[h] = p
    out = []
    for (ci, m), variants in sorted(groups.items()):
        vs = list(variants.items())
        if len(vs) < 2:
            continue
        a = torch.load(vs[0][1])
        for hb, pb in vs[1:]:
            b = torch.load(pb)
            assert a.shape == b.shape and a.dtype == b.dtype
            ab, bb = a.view(torch.int16), b.view(torch.int16)
            diff = (ab != bb)
            nd = int(diff.sum())
            idx = diff.nonzero()[:64]
            fa, fb = a.float(), b.float()
            dvals = (fa - fb)[diff]
            rec = {
                "ci": ci, "m": m, "shape": list(a.shape),
                "dtype": str(a.dtype),
                "n_diff_bytes_halfwords": nd,
                "total_halfwords": ab.numel(),
                "frac": nd / ab.numel(),
                "max_abs_diff": float(dvals.abs().max()) if nd else 0.0,
                "mean_abs_diff": float(dvals.abs().float().mean()) if nd else 0.0,
                "rows_affected": len({int(i) for i in idx[:, 0]}),
                "cols_sample": [[int(r), int(c)] for r, c in idx[:16]],
                "variants": [vs[0][0], hb],
            }
            print(json.dumps(rec), flush=True)
            out.append(rec)
    json.dump(out, open(f"{d}/dump_diff.json", "w"), indent=1)


if __name__ == "__main__":
    main()
