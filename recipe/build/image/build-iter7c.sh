#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build iter7c (DETHASH per-layer hash localization, D-011) on top of the
# pinned iter7b image.  Same pattern as build-iter7b.sh.
# CPU gates (offline, no GPU): (1) awdbg_hash selftest, installs the hooks
# on a toy model, checks well-formed JSONL, hash stability across identical
# forwards, hash sensitivity to a 1e-3 weight perturbation, and the
# execute_model forward counter; (2) import + ast of the patched
# model_runner (the patch itself already asserts its anchor at build time).
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/Dockerfile.iter7c" ] || { echo 'missing Dockerfile.iter7c' >&2; exit 66; }
[ -f "$here/src/awdbg_hash.py" ] || { echo 'missing awdbg_hash module' >&2; exit 66; }
[ -f "$here/src/patch_layer_hashes.py" ] || { echo 'missing dethash patch' >&2; exit 66; }
iter7c='qwen38-flash-dgx:iter7c-20261014'

build_cmd="docker build --network none --pull=false -f '$here/Dockerfile.iter7c' -t '$iter7c' '$here'"
gate1_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' \
  --entrypoint python3 '$iter7c' /selftest/src/awdbg_hash.py"
gate2_cmd="docker run --rm --network none --memory 8g \
  --entrypoint python3 '$iter7c' -c 'import ast, awdbg_hash, vllm.v1.worker.gpu.model_runner as mr, inspect; \
src=inspect.getsourcefile(mr); ast.parse(open(src).read()); \
assert \"DETHASH\" in open(src).read(); print(\"dethash wiring import+ast OK\")'"

if [ "$mode" = --print ]; then
  printf '%s\n' "$build_cmd" "$gate1_cmd" "$gate2_cmd"
  exit 0
fi

eval "$build_cmd"
eval "$gate1_cmd"
eval "$gate2_cmd"

docker image inspect "$iter7c" --format '{{.Id}}'
