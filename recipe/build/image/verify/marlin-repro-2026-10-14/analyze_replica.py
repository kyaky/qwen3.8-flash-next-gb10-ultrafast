#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze the replica server's DETGEMM stream: variant counts per call
index over the ntok==24 prefill forwards (the D-011 target shape M=24).
"""
import collections
import json
import sys


def main():
    path = sys.argv[1]
    fwd, gemm = {}, collections.defaultdict(dict)
    for line in open(path):
        tag, _, js = line.partition(" ")
        if tag == "DETFWD":
            d = json.loads(js)
            if d.get("d", 0) == 0:
                fwd[d["f"]] = d
        elif tag == "DETGEMM":
            d = json.loads(js)
            if d.get("d", 0) != 0:
                continue
            gemm[d["f"]][d["ci"]] = (d["sm"], d["sn"], d["sk"], d["tk"],
                                     d["h"])
    groups = collections.defaultdict(list)
    for f, d in fwd.items():
        if gemm.get(f):
            groups[(d["n"], tuple(sorted(set(g[:4] for g in gemm[f]
                                                  .values()))))].append(f)
    any_flip = False
    for key, fs in sorted(groups.items()):
        if len(fs) < 2:
            continue
        ntok = key[0]
        cis = sorted(set().union(*[set(gemm[f]) for f in fs]))
        div = []
        for ci in cis:
            hs = collections.Counter(gemm[f][ci][4] for f in fs
                                     if ci in gemm[f])
            if len(hs) > 1:
                div.append((ci, dict(hs.most_common())))
        flips = sum(1 for f in fs
                    if any(gemm[f][ci][4] != gemm[fs[0]][ci][4] for ci in cis
                           if ci in gemm[f] and ci in gemm[fs[0]]))
        print(f"ntok={ntok} forwards={len(fs)} cis={len(cis)} "
              f"flipping_forwards={flips} diverging_cis={len(div)}")
        for ci, hist in div[:6]:
            print(f"   ci={ci} shape={gemm[fs[0]][ci][:4]} variants={hist}")
        if div:
            any_flip = True
    print("BIMODAL" if any_flip else "NO DIVERGENCE (all identical forwards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
