#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build iter7g (P2 stage-1 closure instrumentation, D-014 conds. 1+2) on top
# of the pinned iter7d image.  Same pattern as build-iter7d.sh.
# CPU gates (offline, no GPU):
#  (1) the stage-1 closure CPU test (tests/test_stage1_closure_cpu.py): the
#      C2 v2 record world through the gate harness's own checks, the defect
#      classes, the committed-prefix hash book and the AWFP fingerprint
#      layer (policy determinism, emit gating, drift comparison, and the
#      marlin code-state check against the IN-IMAGE patched module);
#  (2) the stage-1b CPU test still passes with the v2 emitter helper
#      (src/test_width_runner_cpu.py - the round-4/5 contract stays green);
#  (3) patch idempotence discipline: restoring the .orig copies and
#      re-applying both patchers reproduces the shipped image bytes
#      EXACTLY, and a second apply FAILS loudly (assert-guarded, no silent
#      no-op path);
#  (4) ast + marker inspection of the patched files in the image.
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
# The CPU tests import the REAL gate harness (driver.py); the repo's
# recipe/tools/gate-stage1b copy is mounted read-only and located through
# GB10_GATE_DIR (both tests honor it; the in-image walk-up cannot see the
# host tree).
gate_dir="$here/../../tools/gate-stage1b"
for f in Dockerfile.iter7g src/patch_c2_fields.py src/patch_fingerprints.py \
         src/awdbg1b_evidence.py src/awfp_fingerprint.py \
         tests/test_stage1_closure_cpu.py src/test_width_runner_cpu.py \
         tests/test_c2_anchor_chain_cpu.py; do
  [ -f "$here/$f" ] || { echo "missing $f" >&2; exit 66; }
done
[ -f "$gate_dir/driver.py" ] || { echo "missing gate harness driver.py at $gate_dir" >&2; exit 66; }
iter7g='qwen38-flash-dgx:iter7g-20261015'
SP=/usr/local/lib/python3.12/dist-packages

build_cmd="docker build --network none --pull=false -f '$here/Dockerfile.iter7g' -t '$iter7g' '$here'"
gate1_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' -v '$here/tests:/selftest/tests:ro' \
  -v '$gate_dir:/selftest/gate:ro' -e GB10_GATE_DIR=/selftest/gate \
  --entrypoint python3 '$iter7g' /selftest/tests/test_stage1_closure_cpu.py"
gate1b_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' -v '$here/tests:/selftest/tests:ro' \
  -v '$gate_dir:/selftest/gate:ro' -e GB10_GATE_DIR=/selftest/gate \
  -e GB10_C2FP_INPUT_DIR=/selftest/tests \
  --entrypoint python3 '$iter7g' /selftest/tests/test_c2_anchor_chain_cpu.py"
# coverage note: inside the image the test's real-input leg (the
# host runs-dir .orig-c2fp walk-up) SKIPS - only the fixture legs
# run here; the real-bytes coverage is gate3's patch reproduction,
# which re-applies the patchers to the preserved .orig copies and
# runs assert_c2_chain_preserved on the genuine scheduler bytes.
gate2_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' -v '$here/tests:/selftest/tests:ro' \
  -v '$gate_dir:/selftest/gate:ro' -e GB10_GATE_DIR=/selftest/gate \
  --entrypoint python3 '$iter7g' /selftest/src/test_width_runner_cpu.py"
gate3_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' \
  --entrypoint python3 '$iter7g' -c '
import hashlib, shutil, subprocess, sys
sp = \"$SP\"
sched = sp + \"/vllm/v1/core/sched/scheduler.py\"
runner = sp + \"/vllm/v1/worker/gpu/model_runner.py\"
spec = sp + \"/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py\"
marlin = sp + \"/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py\"
def md5(p):
    return hashlib.md5(open(p, \"rb\").read()).hexdigest()
shipped = {p: md5(p) for p in (sched, runner, spec, marlin)}
# reproduce the shipped bytes from the preserved pre-patch copies
shutil.copy(sched + \".orig-c2fp\", sched)
shutil.copy(runner + \".orig-c2fp\", runner)
shutil.copy(spec + \".orig-c2fp\", spec)
shutil.copy(marlin + \".orig-awfp\", marlin)
r = subprocess.run([sys.executable, \"/selftest/src/patch_c2_fields.py\",
                    sched, runner, spec], capture_output=True, text=True)
assert r.returncode == 0, r.stderr + r.stdout
r = subprocess.run([sys.executable, \"/selftest/src/patch_fingerprints.py\",
                    marlin, runner], capture_output=True, text=True)
assert r.returncode == 0, r.stderr + r.stdout
for p, h in shipped.items():
    assert md5(p) == h, \"re-applied patch differs from the shipped bytes: \" + p
# a second apply must FAIL loudly (assert-guarded, no silent no-op path)
r = subprocess.run([sys.executable, \"/selftest/src/patch_c2_fields.py\",
                    sched, runner, spec], capture_output=True, text=True)
assert r.returncode != 0, \"second c2_fields apply unexpectedly succeeded\"
r = subprocess.run([sys.executable, \"/selftest/src/patch_fingerprints.py\",
                    marlin, runner], capture_output=True, text=True)
assert r.returncode != 0, \"second fingerprints apply unexpectedly succeeded\"
print(\"stage-1 closure patch idempotence OK\")'"
gate4_cmd="docker run --rm --network none --memory 8g \
  --entrypoint python3 '$iter7g' -c '
import ast
sp = \"$SP\"
files = [
  sp + \"/vllm/v1/core/sched/scheduler.py\",
  sp + \"/vllm/v1/worker/gpu/model_runner.py\",
  sp + \"/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py\",
  sp + \"/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py\",
  sp + \"/awdbg1b_evidence.py\",
  sp + \"/awfp_fingerprint.py\",
]
for p in files:
    ast.parse(open(p).read())
mm = open(files[3]).read()
assert mm.count(\"# qwen38-awfp-begin\") == 1 and mm.count(\"# qwen38-awfp-end\") == 1
assert mm.count(\"# qwen38-moe-align-stable-begin\") == 1
sched = open(files[0]).read()
for marker in (\"_awdbg_c2_field_stashes\", \"_awdbg_commit_capture\",
               \"_awdbg_c2_emit_prep\"):
    assert sched.count(marker) == 1, marker
run = open(files[1]).read()
for marker in (\"_awfp_install\", \"_awfp_policy\", \"_awdbg_worker_emitter\"):
    assert run.count(marker) == 1, marker
spec = open(files[2]).read()
assert spec.count(\"_awdbg_draft_branch\") == 1
print(\"stage-1 closure ast+marker OK\")'"

if [ "$mode" = --print ]; then
  printf '%s\n' "$build_cmd" "$gate1_cmd" "$gate1b_cmd" "$gate2_cmd" "$gate3_cmd" "$gate4_cmd"
  exit 0
fi

eval "$build_cmd"
eval "$gate1_cmd"
eval "$gate1b_cmd"
eval "$gate2_cmd"
eval "$gate3_cmd"
eval "$gate4_cmd"

docker image inspect "$iter7g" --format '{{.Id}}'
