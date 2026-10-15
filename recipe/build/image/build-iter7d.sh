#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build iter7d (moe-align stable-sort shim, D-011 tie-flip fix) on top of the
# pinned iter7c image.  Same pattern as build-iter7c.sh.
# CPU gates (offline, no GPU): (1) the CPU unit test of the canonicalization
# helper against the round-5 reference shim (300 fuzz cases + hand-built +
# gate-off + determinism, tests/test_moe_align_stable_cpu.py); (2) patch
# idempotence - re-applying the patcher to the pristine source reproduces the
# shipped image bytes exactly and a second run is a verified no-op;
# (3) ast + marker + call-site inspection of the patched file in the image
# (the patch itself already asserts its anchor at build time).
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/Dockerfile.iter7d" ] || { echo 'missing Dockerfile.iter7d' >&2; exit 66; }
[ -f "$here/src/patch_moe_align_stable.py" ] || { echo 'missing moe-align-stable patch' >&2; exit 66; }
[ -f "$here/tests/test_moe_align_stable_cpu.py" ] || { echo 'missing moe-align-stable cpu test' >&2; exit 66; }
iter7d='qwen38-flash-dgx:iter7d-20261014'

build_cmd="docker build --network none --pull=false -f '$here/Dockerfile.iter7d' -t '$iter7d' '$here'"
gate1_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' -v '$here/tests:/selftest/tests:ro' \
  --entrypoint python3 '$iter7d' \
  /selftest/tests/test_moe_align_stable_cpu.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
gate2_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' \
  --entrypoint python3 '$iter7d' -c '
import hashlib, shutil, subprocess, sys
sp = \"/usr/local/lib/python3.12/dist-packages\"
mm = sp + \"/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py\"
orig = mm + \".orig-alignstable\"
patcher = \"/selftest/src/patch_moe_align_stable.py\"
h0 = hashlib.md5(open(mm, \"rb\").read()).hexdigest()
shutil.copy(orig, mm)
r = subprocess.run([sys.executable, patcher, sp], capture_output=True, text=True)
assert r.returncode == 0, r.stderr + r.stdout
assert \"applied OK\" in r.stdout, r.stdout
h1 = hashlib.md5(open(mm, \"rb\").read()).hexdigest()
assert h1 == h0, \"re-applied patch differs from shipped image state\"
r = subprocess.run([sys.executable, patcher, sp], capture_output=True, text=True)
assert r.returncode == 0, r.stderr + r.stdout
assert \"already applied\" in r.stdout, r.stdout
h2 = hashlib.md5(open(mm, \"rb\").read()).hexdigest()
assert h2 == h0, \"idempotence second run changed bytes\"
shutil.copy(orig, mm)
print(\"moe-align-stable idempotence OK\")'"
gate3_cmd="docker run --rm --network none --memory 8g \
  --entrypoint python3 '$iter7d' -c '
import ast
mm = \"/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py\"
s = open(mm).read()
ast.parse(s)
assert s.count(\"# qwen38-moe-align-stable-begin\") == 1
assert s.count(\"# qwen38-moe-align-stable-end\") == 1
assert \"sorted_token_ids = _q38_moe_align_stable(\" in s
assert \".orig-alignstable\" not in s
print(\"moe-align-stable ast+anchor OK\")'"

if [ "$mode" = --print ]; then
  printf '%s\n' "$build_cmd" "$gate1_cmd" "$gate2_cmd" "$gate3_cmd"
  exit 0
fi

eval "$build_cmd"
eval "$gate1_cmd"
eval "$gate2_cmd"
eval "$gate3_cmd"

docker image inspect "$iter7d" --format '{{.Id}}'
