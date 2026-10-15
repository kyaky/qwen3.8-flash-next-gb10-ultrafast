#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Runner for the GB10 P2 stage-1 kernel-level stochastic correctness gate
# (R-031, astra closure-alignment task-mv8oapbk).
#
# Runs recipe/build/image/tests/test_blockverifier_gate.py inside a THROWAWAY
# --gpus all container built from the pinned image (the running qwen38-flash
# server is never touched; source reads there are read-only docker exec).
#
# Usage:
#   run_blockverifier_gate.sh [IMAGE] [RESULTS_JSON] [--quick]
#   IMAGE        default qwen38-flash-dgx:iter7f-20261015
#                (sampler files sha-identical in iter7d-20261014)
#   RESULTS_JSON default $GB10_WORKSPACE/agents/
#                glm-5.3-flash-integrator/runs/2026-10-15-blockverifier-gate/
#                blockverifier_gate_results.json
#
# Exit code 0 = gate passed (all checks ok AND every sensitivity control
# rejected); nonzero = gate failure.
set -euo pipefail

IMAGE="${1:-qwen38-flash-dgx:iter7f-20261015}"
[ $# -gt 0 ] && shift
RESULTS="${1:-$GB10_AGENT_DIR/runs/2026-10-15-blockverifier-gate/blockverifier_gate_results.json}"
# a non-flag second argument IS the results path; consume it so it does not
# leak into the test's own flags (arg-parsing bug found on the rerun)
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then shift; fi
EXTRA_ARGS="${*:-}"

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$RESULTS")"
mkdir -p "$RUN_DIR"

echo "image: $IMAGE"
echo "results: $RESULTS"
docker run --rm --gpus all \
  --entrypoint /bin/bash \
  -v "$TEST_DIR":/gate/tests:ro \
  -v "$RUN_DIR":/work/results \
  -e TRITON_CACHE_DIR=/tmp/triton_cache_gate \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e "GB10_GATE_IMAGE=$IMAGE" \
  "$IMAGE" \
  -lc "cd /gate/tests && python3 -I test_blockverifier_gate.py \
        --json-out /work/results/$(basename "$RESULTS") $EXTRA_ARGS"
