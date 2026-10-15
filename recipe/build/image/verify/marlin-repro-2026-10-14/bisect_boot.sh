#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One bisect boot of the marlin replica: reboot with MARLIN_SHIM=<mode>,
# wait for readiness, fire N identical requests, report.
# usage: bisect_boot.sh <mode> <tag> <n_requests> [prompt_words]
set -euo pipefail
R=$GB10_AGENT_DIR/runs/2026-10-14-marlin-repro
MODE="${1:-}"
TAG="${2:?tag}"
N="${3:-30}"
PW="${4:-0}"

docker rm -f marlin-replica >/dev/null 2>&1 || true
docker run -d --name marlin-replica --gpus all --ipc=host --shm-size 16g \
  -p 18800:8000 \
  -v "$R"/tiny-ckpt:/model:ro \
  -v $HOME/models/ple-table-fp8:/ple-table:ro \
  -v "$R"/logs:/detout \
  -v "$R"/repro/shim:/bisect:ro \
  -e MARLIN_SHIM="$MODE" -e MARLIN_FORCE_BLOCK="${MARLIN_FORCE_BLOCK:-}" \
  -e MARLIN_SHIM_LOG="/detout/shim_$TAG.jsonl" \
  -e MARLIN_SHIM_LOG_DIR=/detout \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=8 -e VLLM_PLE_MMAP_PREWARM=0 \
  -e VLLM_PLE_MMAP_PREFETCH=0 -e VLLM_PLE_MMAP_MADV_RANDOM=1 \
  -e VLLM_PLE_MMAP_FAST_PATH=1 -e VLLM_PLE_MMAP_FAST_ROWS=16 \
  -e VLLM_PLE_MMAP_FAST_MAX_ROWS=262144 \
  -e VLLM_PLE_MMAP_CHUNK=8 -e VLLM_PLE_MMAP_STATS_SEC=30 \
  -e VLLM_PLE_MMAP_DIR=/ple-table \
  -e QWEN38NEXT_LOW_LATENCY_GEMM=1 -e QWEN38NEXT_LLG_PDL=0 \
  -e VLLM_VERIFY_TOPK_TRITON=1 -e VLLM_KEEP_DRAFT_BLOCKS=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_FP8_HYBRID=1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e VLLM_QSA_DET=1 -e VLLM_LOAD_CLONE=1 -e VLLM_MOE_NAME_INDEX=1 \
  -e VLLM_LOAD_EMBED_CHUNK=1 -e VLLM_ADAPTIVE_WIDTH_MODE=shadow \
  -e DET_HASH_LAYERS=1 -e DET_HASH_MIN_LAYERS=4 \
  -e DET_HASH_GEMM_PROBE=1 -e DET_HASH_FILE="/detout/det_$TAG.jsonl" \
  --entrypoint /bin/bash \
  qwen38-flash-dgx:iter7c-20261014 /bisect/boot.sh >/dev/null

for i in $(seq 1 240); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:18800/v1/models 2>/dev/null || true)
  [ "$code" = "200" ] && break
  if docker logs marlin-replica 2>&1 | grep -qE "Traceback|EngineCore encountered|DETHASH-FATAL"; then
    echo "BOOT_FAILED ($TAG)"; docker logs marlin-replica 2>&1 | tail -5; exit 1
  fi
  sleep 3
done
[ "$code" = "200" ] || { echo "BOOT_TIMEOUT ($TAG)"; exit 1; }
echo "boot OK: $TAG (mode='$MODE')"

python3 - "$N" "$PW" <<'PYEOF'
import json, sys, time, urllib.request
n, pw = int(sys.argv[1]), int(sys.argv[2])
WORDS = ("the quick brown fox jumps over the lazy dog while clouds gather "
         "over silent valleys and rivers carry stones").split()
PROMPT = (" ".join(WORDS[:pw]) + " " if pw else "") + \
    "Write a Python function that returns the median of three numbers. " \
    "Answer with code only, no explanation, and keep it short please."
for i in range(n):
    body = json.dumps({"model": "qwen",
                       "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 1, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request("http://localhost:18800/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    print(f"req{i}: ptok={d['usage'].get('prompt_tokens')}", end=" ")
print()
PYEOF
echo "probe done: $TAG"
