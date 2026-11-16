#!/usr/bin/env bash
# Start the model server (vLLM, OpenAI-compatible API on port 8000).
# Memory: --gpu-memory-utilization 0.90 of a 24 GB card keeps the process under ~21.6 GB.
# On a larger card this script refuses to start unless you cap it, so that a run can never
# depend on more than 24 GB.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-12288}"
MAX_SEQS="${MAX_SEQS:-16}"

TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
# fraction of this card that equals 21.6 GB (0.90 x 24 GB), never above 0.90
FRACTION=$(python3 -c "t=$TOTAL_MIB; print(f'{min(0.90, 22118.0/t):.3f}')")
echo "GPU memory ${TOTAL_MIB} MiB -> using fraction ${FRACTION} (cap 21.6 GB)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # one GPU only

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$FRACTION" \
  --limit-mm-per-prompt '{"image": 3}' \
  --seed 42 \
  "$@"
