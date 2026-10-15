#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the iter6e image (floor iter6d + deterministic QSA top-k stage) without
# touching the donor's build.sh. Preflight pins the five vendored kernel-det
# sources to the sha256 values recorded in the source Dockerfile ADD --checksum
# lines (source: kernel-det vendored sources @e0ef69d4f5575dad00d34e05479eaf4c6547bace).
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/Dockerfile.iter6e" ] || { echo 'missing Dockerfile.iter6e' >&2; exit 66; }
[ -f "$here/tests/test_iter6e_image.py" ] || { echo 'missing iter6e gate test' >&2; exit 66; }
iter6e='qwen38-flash-dgx:iter6e-20261012'

if ! (cd "$here/src/kernel-det" && sha256sum --check --status <<'SHA256'
138cacfc5eb117f0922d53c88727e4d0dc26dcfb246c3d401fc280cfc726cc71  build_det.py
b103fbeaf7589b9468471142ad0b30012a076f93d20ba11fc5ff6dcb1ecd32a6  bindings_det.cpp
19e1d53425ea9a839445722fd1dac1c41727128eebdce84508c1bfb8592afecf  topk_det.cu
16939700ae389750782ff5c0d5b9caef59aa0ff8b869b64ec94fa72c814910ee  torch_utils.h
b4ef9ce298d43d6c0e6db9fcca451df20815b2cfe33791919c1ad9c0e84f0ba7  persistent_topk.cuh
SHA256
); then
  echo 'kernel-det source sha256 mismatch; refusing build' >&2
  exit 65
fi

build_cmd="docker build --network none --pull=false -f '$here/Dockerfile.iter6e' -t '$iter6e' '$here'"
gate_cmd="docker run --rm --network none --memory 8g \
  -v '$here/tests/test_iter6e_image.py:/tmp/t.py:ro' \
  -v '$here/src:/selftest/src:ro' \
  --entrypoint python3 '$iter6e' /tmp/t.py"

if [ "$mode" = --print ]; then
  printf '%s\n' "$build_cmd" "$gate_cmd"
  exit 0
fi

eval "$build_cmd"
eval "$gate_cmd"

docker image inspect "$iter6e" --format '{{.Id}}'
