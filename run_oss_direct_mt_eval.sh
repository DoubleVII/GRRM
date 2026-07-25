#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL_PATH="${MODEL_PATH:-/home/zfs01/yangs/LLM/openai/gpt-oss-120b}"
DATA_IDS="${DATA_IDS:-seedx_challenge_zhen}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-results/oss_direct_mt_eval.json}"

DATA_IDS=seedx_challenge_zhen,seedx_challenge_enzh,wmt23_zh_en,wmt24pp_en_zh


.venv/bin/python -m eval.run_oss_direct_mt_eval \
  --data_id "${DATA_IDS}" \
  --model_path "${MODEL_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --max_samples "${MAX_SAMPLES}" \
  --reasoning_effort medium \
  --temperature 0.3 \
  --top_p 0.8 \
  --max_new_tokens 4096 \
  --retry 3 \
  --gpu_memory_utilization 0.9 \
  --max_model_len 32768
