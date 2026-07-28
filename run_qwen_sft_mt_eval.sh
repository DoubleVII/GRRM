#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3}"
export WANDB_MODE="${WANDB_MODE:-offline}"

: "${MODEL_PATH:?Set MODEL_PATH to the trained Qwen checkpoint}"
METHOD="${METHOD:-scd}"
MODEL_LABEL="${MODEL_LABEL:-$(basename "${MODEL_PATH}")}"
EVALUATOR_MODEL_PATH="${EVALUATOR_MODEL_PATH:-/home/zfs01/yangs/LLM/openai/gpt-oss-120b}"
DATA_IDS="${DATA_IDS:-seedx_challenge_zhen,seedx_challenge_enzh,wmt23_zh_en,wmt24pp_en_zh}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RUNS="${RUNS:-4}"
SAMPLING_N="${SAMPLING_N:-4}"
OUTPUT_PATH="${OUTPUT_PATH:-results/qwen_sft_mt_eval.${MODEL_LABEL}.${METHOD}.json}"

.venv/bin/python -m eval.run_qwen_sft_mt_eval \
  --method "${METHOD}" \
  --model_path "${MODEL_PATH}" \
  --model_label "${MODEL_LABEL}" \
  --evaluator_model_path "${EVALUATOR_MODEL_PATH}" \
  --data_id "${DATA_IDS}" \
  --output_path "${OUTPUT_PATH}" \
  --max_samples "${MAX_SAMPLES}" \
  --sampling_n "${SAMPLING_N}" \
  --runs "${RUNS}" \
  --gpu_memory_utilization 0.9 \
  --evaluator_gpu_memory_utilization 0.9 \
  --max_model_len 32768
