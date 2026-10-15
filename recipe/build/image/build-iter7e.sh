#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build iter7e-det (THROWAWAY: torch global deterministic-algorithms switch,
# torch-determinism lane 2026-10-14) on top of the local iter7d image.
# Same pattern as build-iter7d.sh.
# Offline CPU gates: (1) the .pth import hook arms the switch with
# warn_only=True in a plain interpreter (site initialization order proof);
# (2) the image ENV carries the cuBLAS workspace config + markers;
# (3) the bootstrap module stays importable and ast-clean.
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/Dockerfile.iter7e" ] || { echo 'missing Dockerfile.iter7e' >&2; exit 66; }
[ -f "$here/src/determinism_bootstrap.py" ] || { echo 'missing determinism_bootstrap.py' >&2; exit 66; }
[ -f "$here/src/zz_determinism_bootstrap.pth" ] || { echo 'missing zz_determinism_bootstrap.pth' >&2; exit 66; }
iter7e='qwen38-flash-dgx:iter7e-det-20261014'

build_cmd="docker build --network none --pull=false -f '$here/Dockerfile.iter7e' -t '$iter7e' '$here'"
gate1_cmd="docker run --rm --network none --memory 8g --entrypoint python3 '$iter7e' -c '
import torch
assert torch.are_deterministic_algorithms_enabled(), \"pth hook did not arm the switch\"
assert torch._C._get_deterministic_algorithms_warn_only(), \"warn_only not armed\"
assert torch.backends.cudnn.deterministic, \"cudnn.deterministic not armed\"
assert torch.backends.cudnn.benchmark is False, \"cudnn.benchmark not pinned\"
print(\"determinism .pth gate OK (warn_only=True)\")'"
gate2_cmd="docker image inspect '$iter7e' --format '{{join .Config.Env \"\n\"}}' | grep -E '^(CUBLAS_WORKSPACE_CONFIG=:16:8|TORCH_USE_DETERMINISTIC_ALGORITHMS=1|DETERM_BOOTSTRAP=1)$'"
gate3_cmd="docker run --rm --network none --memory 8g --entrypoint python3 '$iter7e' -c '
import determinism_bootstrap, hashlib
h = hashlib.sha256(open(determinism_bootstrap.__file__, \"rb\").read()).hexdigest()
print(\"determinism_bootstrap import OK sha256\", h)'"

if [ "$mode" = --print ]; then
  printf '%s\n' "$build_cmd" "$gate1_cmd" "$gate2_cmd" "$gate3_cmd"
  exit 0
fi

eval "$build_cmd"
eval "$gate1_cmd"
eval "$gate2_cmd"
eval "$gate3_cmd"

docker image inspect "$iter7e" --format '{{.Id}}'
