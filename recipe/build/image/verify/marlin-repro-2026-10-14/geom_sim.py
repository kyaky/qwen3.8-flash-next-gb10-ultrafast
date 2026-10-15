#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Simulate marlin_moe_wna16 DP+two-tile-SK slice geometry (marlin_template.h
355-631, main loop 1818-1958) for the layer-3 shapes on GB10 (48 SMs).

Determines, per launch geometry, whether any slice has slice_count > 1 (the
only case that uses locks[workspace] and the fp32 C_tmp partial buffer) and
whether the locks_off slot assignment can collide between slices.
"""
import math


def div_ceil(a, b):
    return -(-a // b)


def simulate(size_m, top_k, size_n, size_k, moe_block_size, nttp, sms=48,
             thread_k=64, thread_n=128, threads=128, blocks_per_sm=3,
             group_size=128, verbose_blocks=()):
    grid = sms * blocks_per_sm
    thread_m_blocks = div_ceil(moe_block_size, 16)
    m_block_size_8 = moe_block_size == 8
    assert thread_m_blocks == 1 and m_block_size_8
    parallel = nttp // moe_block_size
    k_tiles = size_k // 16 // (thread_k // 16)
    n_tiles = size_n // 16 // (thread_n // 16)
    global_mn_tiles = parallel * n_tiles
    part2_mn_tiles = global_mn_tiles
    part1_mn_iters = 0
    if global_mn_tiles > grid:
        part2_mn_tiles = global_mn_tiles % grid
        if part2_mn_tiles * 3 <= grid:
            part2_mn_tiles += grid
        part1_mn_iters = (global_mn_tiles - part2_mn_tiles) // grid
    iters = div_ceil(k_tiles * part2_mn_tiles, grid)
    group_blocks = group_size // 16
    if group_blocks >= thread_k // 16:
        iters = (group_blocks // (thread_k // 16)) * div_ceil(
            iters, group_blocks // (thread_k // 16))

    # locks_off init
    if part2_mn_tiles >= grid:
        locks_off_init = lambda b: b
        in_part2_branch = "A"
    else:
        locks_off_init = lambda b: (iters * b) // k_tiles - 1
        in_part2_branch = "B"

    slices = []  # (block, seq, mn_tile=col_par, row, iters, count, idx, lock)
    for b in range(grid):
        lo = locks_off_init(b)
        scp = b  # part1 start: slice_col_par = blockIdx.x
        row = 0
        in_part2 = False
        p1 = part1_mn_iters
        seq = 0
        first_init = True
        while True:
            if not in_part2:
                if p1 > 0:
                    # init_part1_slice: full mn-tile, slice_count=1
                    p1 -= 1
                    slices.append((b, seq, scp, 0, k_tiles, 1, 0, None,
                                   "part1"))
                    seq += 1
                    scp += grid
                    continue
                # transition to part2
                in_part2 = True
                scp = (iters * b) // k_tiles
                row = (iters * b) % k_tiles
            # init_part2_slice
            si = iters * (b + 1) - (k_tiles * scp + row)
            if si < 0 or scp >= part2_mn_tiles:
                break
            if row + si > k_tiles:
                si = k_tiles - row
            if si == 0:
                break
            count = 1
            idx = 0
            col_first = iters * div_ceil(k_tiles * scp, iters)
            if col_first <= k_tiles * (scp + 1):
                col_off = col_first - k_tiles * scp
                count = div_ceil(k_tiles - col_off, iters)
                if col_off > 0:
                    count += 1
                delta_first = iters * b - col_first
                if delta_first < 0 or (col_off == 0 and delta_first == 0):
                    idx = count - 1
                else:
                    idx = count - 1 - delta_first // iters
                    if col_off > 0:
                        idx -= 1
            used = None
            if in_part2_branch == "A":
                if count > 1 and idx == count - 1:
                    lo += 1
                    used = lo
            else:
                lo += 1
                used = lo
            if count > 1:
                slices.append((b, seq, scp, row, si, count, idx, used,
                               "part2"))
            seq += 1
            # advance
            scp += 1
            row = 0
            first_init = False
            if scp >= part2_mn_tiles:
                break
        # end block

    multi = [s for s in slices if s[5] > 1]
    # sanity: blocks working the same (col_par, row-window) slice must agree
    # on (count, idx); collect slice groups by (col_par, row)
    groups = {}
    for s in slices:
        if s[5] > 1:
            groups.setdefault((s[2], s[3]), set()).add((s[5], s[6], s[7]))
    inconsistent = {k: v for k, v in groups.items() if len(v) > 1}
    locks_used = sorted({s[7] for s in multi if s[7] is not None})
    print(f"size_m={size_m} top_k={top_k} n={size_n} k={size_k} "
          f"moe_block={moe_block_size} nttp={nttp} | grid={grid} "
          f"branch={in_part2_branch} parallel={parallel} n_tiles={n_tiles} "
          f"mn_tiles={global_mn_tiles} part2={part2_mn_tiles} "
          f"part1_iters={part1_mn_iters} iters={iters}")
    print(f"  multi-block slices: {len(multi)} "
          f"(blocks involved: {len({s[0] for s in multi})})")
    if multi:
        cs = sorted({s[5] for s in multi})
        print(f"  slice_count values: {cs}; distinct locks_off used: "
              f"{len(locks_used)} range [{locks_used[0]},{locks_used[-1]}]")
    for b in verbose_blocks:
        bs = [s for s in slices if s[0] == b]
        print(f"  block {b}: {len(bs)} slices; first few: "
              f"{[(s[2], s[3], s[4], s[5], s[6], s[7]) for s in bs[:4]]}")
    return len(multi), groups, inconsistent


if __name__ == "__main__":
    import sys
    # repro GEMM1: nttp=1416
    simulate(24, 10, 1280, 2560, 8, 1416, verbose_blocks=(0, 1, 2, 143))
    # repro GEMM2
    simulate(240, 1, 2560, 640, 8, 1416, verbose_blocks=())
    # geometry sweep over plausible serving nttp values
    for nttp in range(1408, 1921, 8):
        m1, _, _ = simulate(24, 10, 1280, 2560, 8, nttp)
        if m1 == 0:
            print(f"  ^ nttp={nttp}: NO multi-block slices for GEMM1")
