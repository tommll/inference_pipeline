#!/usr/bin/env bash
# Start llama-server tuned for this machine (ThinkPad X1 Carbon Gen 11, i7-1365U).
# Flag choices are explained in the research brief; the short version:
#   -t 8            measured bandwidth peaked at 8 threads, got noisier at 12
#   -c 16384        Qwen3.5 advertises a huge context; allocating it would eat all RAM
#   --cache-type-*  q8_0 KV cache roughly halves KV memory for negligible quality loss
#
# Not enabled: --load-mode mlock (the old --mlock). It pins weights so they are
# never swapped out, but it failed on this box even with ulimit -l at ~3.9 GB,
# and with 15 GiB free the page cache keeps the weights resident anyway. Add it
# back only if you see swapping, and never with the 17 GB MoE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/llamacpp/llama-b10785"
MODEL="${MODEL:-$ROOT/models/Qwen3.5-4B-Q4_K_M.gguf}"
PORT="${PORT:-8080}"

# VULKAN=1 offloads to the Iris Xe iGPU. Measured on this machine with
# Qwen3.5-4B-Q4_K_M (llama-bench, 8 threads):
#
#              prefill (pp512)   decode (tg128)
#   CPU           14.1 tok/s       5.82 tok/s
#   Vulkan        38.5 tok/s       3.58 tok/s
#
# The iGPU shares the same memory controller, so it cannot help bandwidth-bound
# decode - and it is actually worse at it. It nearly triples compute-bound
# prefill. Break-even is at roughly 2.4 prompt tokens per generated token:
# use VULKAN=1 for summarising or RAG (long input, short answer), and the CPU
# default for chat (short input, long answer).
GPU_ARGS=()
if [[ "${VULKAN:-0}" == "1" ]]; then
  GPU_ARGS=(-ngl 99)
else
  # -ngl 0 is explicit: the Vulkan backend is present in this build directory
  # and will be discovered, so pin layers to the CPU rather than rely on a default.
  GPU_ARGS=(-t 8 -ngl 0)
fi

if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"

exec "$BIN/llama-server" \
  -m "$MODEL" \
  "${GPU_ARGS[@]}" \
  -c 16384 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -b 1024 -ub 512 \
  --host 127.0.0.1 \
  --port "$PORT" \
  "$@"
