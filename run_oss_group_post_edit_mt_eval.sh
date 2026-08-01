#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL_PATH="${MODEL_PATH:-/home/zfs01/yangs/LLM/openai/gpt-oss-120b}"
DATA_IDS="${DATA_IDS:-seedx_challenge_zhen,seedx_challenge_enzh,wmt23_zh_en,wmt24pp_en_zh}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLING_N="${SAMPLING_N:-4}"
RUNS="${RUNS:-4}"
OUTPUT_PATH="${OUTPUT_PATH:-results/oss_group_post_edit_mt_eval.n${SAMPLING_N}.json}"

.venv/bin/python -m eval.run_oss_group_post_edit_mt_eval \
  --data_id "${DATA_IDS}" \
  --model_path "${MODEL_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --max_samples "${MAX_SAMPLES}" \
  --sampling_n "${SAMPLING_N}" \
  --reasoning_effort medium \
  --sampling_temperature 0.8 \
  --sampling_top_p 0.95 \
  --sampling_max_tokens 4096 \
  --post_edit_temperature 0.3 \
  --post_edit_top_p 0.8 \
  --post_edit_max_tokens 4096 \
  --retry 3 \
  --runs "${RUNS}" \
  --gpu_memory_utilization 0.85 \
  --max_model_len 32768
