#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build iter7b (P2 stage-1c armed wiring) on top of the pinned iter7a image.
# Donor build.sh untouched; same pattern as build-iter7a.sh.
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/Dockerfile.iter7b" ] || { echo 'missing Dockerfile.iter7b' >&2; exit 66; }
[ -f "$here/src/vllm_adaptive_width.py" ] || { echo 'missing controller module' >&2; exit 66; }
[ -f "$here/src/patch_adaptive_width_armed.py" ] || { echo 'missing armed wiring patch' >&2; exit 66; }
[ -f "$here/src/patch_width_runner.py" ] || { echo 'missing stage-1b runner width patch' >&2; exit 66; }
[ -f "$here/src/awdbg1b_evidence.py" ] || { echo 'missing AWDBG1B evidence helper' >&2; exit 66; }
[ -f "$here/src/test_adaptive_width_cpu.py" ] || { echo 'missing CPU gate' >&2; exit 66; }
iter7b='qwen38-flash-dgx:iter7b-20261013'

build_cmd="docker build --network none --pull=false -f '$here/Dockerfile.iter7b' -t '$iter7b' '$here'"
gate_cmd="docker run --rm --network none --memory 8g \
  -v '$here/tests/test_iter6e_image.py:/tmp/t6e.py:ro' \
  -v '$here/src:/selftest/src:ro' \
  --entrypoint python3 '$iter7b' /tmp/t6e.py"
gate2_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' \
  --entrypoint python3 '$iter7b' /selftest/src/test_adaptive_width_cpu.py"

if [ "$mode" = --print ]; then
  printf '%s\n' "$build_cmd" "$gate_cmd" "$gate2_cmd"
  exit 0
fi

eval "$build_cmd"
eval "$gate_cmd"
eval "$gate2_cmd"

docker image inspect "$iter7b" --format '{{.Id}}'
