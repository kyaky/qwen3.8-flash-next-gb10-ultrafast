#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build iter7a (P2 stage-1a shadow wiring) on top of the pinned iter6e image.
# Donor build.sh untouched; same pattern as build-iter6e.sh.
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/Dockerfile.iter7a" ] || { echo 'missing Dockerfile.iter7a' >&2; exit 66; }
[ -f "$here/src/vllm_adaptive_width.py" ] || { echo 'missing controller module' >&2; exit 66; }
[ -f "$here/src/patch_adaptive_width.py" ] || { echo 'missing wiring patch' >&2; exit 66; }
[ -f "$here/src/test_adaptive_width_cpu.py" ] || { echo 'missing CPU gate' >&2; exit 66; }
iter7a='qwen38-flash-dgx:iter7a-20261013'

build_cmd="docker build --network none --pull=false -f '$here/Dockerfile.iter7a' -t '$iter7a' '$here'"
gate_cmd="docker run --rm --network none --memory 8g \
  -v '$here/tests/test_iter6e_image.py:/tmp/t6e.py:ro' \
  -v '$here/src:/selftest/src:ro' \
  --entrypoint python3 '$iter7a' /tmp/t6e.py"
gate2_cmd="docker run --rm --network none --memory 8g \
  -v '$here/src:/selftest/src:ro' \
  --entrypoint python3 '$iter7a' /selftest/src/test_adaptive_width_cpu.py"

if [ "$mode" = --print ]; then
  printf '%s\n' "$build_cmd" "$gate_cmd" "$gate2_cmd"
  exit 0
fi

eval "$build_cmd"
eval "$gate_cmd"
eval "$gate2_cmd"

docker image inspect "$iter7a" --format '{{.Id}}'
