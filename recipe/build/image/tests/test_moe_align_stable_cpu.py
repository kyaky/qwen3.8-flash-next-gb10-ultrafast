#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test for the moe-align stable-sort shim (D-011 tie-flip fix).

Loads the exact helper text the patcher appends to
vllm/model_executor/layers/fused_moe/experts/marlin_moe.py (marker block
"qwen38-moe-align-stable"), executes it against plain torch on CPU, and checks
the canonicalization against an independent reference implementation (the
per-group host loop measured at 0/30 divergence in the round-5 serving run).

Invariants checked on every case:
  - group membership preserved (per-group valid pair multiset identical);
  - order canonical (within each contiguous expert group the valid pair ids
    ascend);
  - sentinel slots (value == numel) stay put;
  - idempotent (applying the shim to its own output is a no-op);
  - byte-equal to the reference shim on kernel-conformant align outputs
    (valid ids lead each group's segment, sentinels trail - the layout the
    align kernel produces).

usage: test_moe_align_stable_cpu.py [path-to-marlin_moe.py]
"""
import os
import random
import sys

import torch

mm_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/experts/marlin_moe.py"
)
src = open(mm_path).read()
begin = src.index("# qwen38-moe-align-stable-begin")
end = src.index("# qwen38-moe-align-stable-end")
block = src[begin:end]

env = os.environ
env["VLLM_MOE_ALIGN_STABLESORT"] = "1"


def load_helper():
    ns = {"torch": torch}
    exec(compile(block, mm_path, "exec"), ns)  # noqa: S102 - repo-owned source
    return ns["_q38_moe_align_stable"]


canon = load_helper()


def reference(sorted_ids, expert_ids, bs, numel):
    """The round-5 measured shim, generalized: sort the leading valid entries
    of every contiguous expert run ascending, leave sentinels in place."""
    ei = expert_ids.tolist()
    h = sorted_ids.tolist()
    out = list(h)
    nb = len(ei)
    i = 0
    while i < nb:
        j = i
        while j + 1 < nb and ei[j + 1] == ei[i]:
            j += 1
        seg = h[i * bs : (j + 1) * bs]
        v = sum(1 for x in seg if x < numel)
        if v > 1:
            out[i * bs : i * bs + v] = sorted(seg[:v])
        i = j + 1
    return torch.tensor(out, dtype=sorted_ids.dtype)


def make_align_output(rng, num_experts, bs, numel):
    """Build a kernel-conformant align output: per-expert valid counts, each
    group's valid ids in random arrival order at the group's leading slots,
    sentinel (numel) padding to block multiples, then the -1 expert tail."""
    counts = [rng.randrange(0, 3 * bs + 1) for _ in range(num_experts)]
    total = sum((c + bs - 1) // bs * bs for c in counts)
    tail_blocks = rng.randrange(0, 3)  # expert_ids -1 tail past the pad
    nb = total // bs + tail_blocks
    sorted_ids = torch.full((nb * bs,), numel, dtype=torch.int32)
    expert_ids = torch.full((nb,), -1, dtype=torch.int32)
    pool = list(range(numel))
    rng.shuffle(pool)
    pos = 0
    next_id = 0
    for e, c in enumerate(counts):
        blocks = (c + bs - 1) // bs
        for b in range(blocks):
            expert_ids[pos // bs + b] = e
        ids = pool[next_id : next_id + c]
        next_id += c
        rng.shuffle(ids)
        for k, pair in enumerate(ids):
            sorted_ids[pos + k] = pair
        pos += blocks * bs
    return sorted_ids, expert_ids


def check_case(sorted_ids, expert_ids, bs, numel, tag):
    before = sorted_ids.clone()
    out = canon(sorted_ids, expert_ids, bs, numel)
    ref = reference(before, expert_ids, bs, numel)
    assert torch.equal(out, ref), f"{tag}: mismatch vs reference shim"
    assert torch.equal(out, canon(out, expert_ids, bs, numel)), (
        f"{tag}: not idempotent"
    )
    valid = before != numel
    assert torch.equal(valid, out != numel), f"{tag}: sentinel slots moved"
    assert torch.equal(
        before[valid].sort().values, out[out != numel].sort().values
    ), f"{tag}: valid multiset changed"
    nb = expert_ids.numel()
    ei = expert_ids.tolist()
    a = out.view(nb, bs).tolist()
    i = 0
    while i < nb:
        j = i
        while j + 1 < nb and ei[j + 1] == ei[i]:
            j += 1
        flat = [x for blk in a[i : j + 1] for x in blk]
        vals = [x for x in flat if x != numel]
        assert vals == sorted(vals), f"{tag}: group {ei[i]} not canonical"
        i = j + 1
    return out


# 1. Hand-built case: uneven groups, empty experts, exact block multiples,
#    single-entry groups, and the -1 sentinel tail.
rng = random.Random(20261014)
bs = 4
numel = 14
groups = [(0, 5), (1, 0), (2, 8), (3, 1)]  # (expert, valid count)
total = sum((c + bs - 1) // bs * bs for _, c in groups)
nb = total // bs + 2
sorted_ids = torch.full((nb * bs,), numel, dtype=torch.int32)
expert_ids = torch.full((nb,), -1, dtype=torch.int32)
pool = list(range(numel))
rng.shuffle(pool)
pos = 0
next_id = 0
for e, c in groups:
    blocks = (c + bs - 1) // bs
    for b in range(blocks):
        expert_ids[pos // bs + b] = e
    ids = pool[next_id : next_id + c]
    next_id += c
    rng.shuffle(ids)
    for k, pair in enumerate(ids):
        sorted_ids[pos + k] = pair
    pos += blocks * bs
check_case(sorted_ids, expert_ids, bs, numel, "hand-built")

# 2. Gate off -> the helper must be the identity.
env["VLLM_MOE_ALIGN_STABLESORT"] = "0"
canon_off = load_helper()
assert torch.equal(canon_off(sorted_ids, expert_ids, bs, numel), sorted_ids), (
    "gate-off call changed the tensor"
)
env["VLLM_MOE_ALIGN_STABLESORT"] = "1"
canon = load_helper()

# 3. Fuzz against the reference shim.
cases = 0
for _ in range(300):
    num_experts = rng.randrange(1, 9)
    bs = rng.randrange(1, 9)
    m = rng.randrange(0, 25)
    topk = rng.randrange(1, 5)
    numel = m * topk
    if numel == 0:
        sorted_ids = torch.empty((0,), dtype=torch.int32)
        expert_ids = torch.empty((0,), dtype=torch.int32)
        out = canon(sorted_ids, expert_ids, bs, numel)
        assert out.numel() == 0
        cases += 1
        continue
    s, e = make_align_output(rng, num_experts, bs, numel)
    check_case(s, e, bs, numel, f"fuzz E={num_experts} bs={bs} numel={numel}")
    cases += 1

# 4. Determinism: identical inputs give identical outputs across repeats.
s, e = make_align_output(rng, 6, 8, 80)
first = canon(s, e, 8, 80)
for _ in range(20):
    assert torch.equal(canon(s, e, 8, 80), first), "nondeterministic repeat"

print(f"test_moe_align_stable_cpu: OK ({cases} fuzz cases + hand-built + "
      "gate-off + determinism)")
