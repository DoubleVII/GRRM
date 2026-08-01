#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

DATA_PATH="${DATA_PATH:-/home/nfs06/yangs/data/parquet_data/mt_distill_data/tower_zhen.raw.parquet}"
MODEL_PATH="${MODEL_PATH:-/home/zfs01/yangs/LLM/openai/gpt-oss-120b}"
PROMPT_TYPE="${PROMPT_TYPE:-fixed_4}"
MAX_CANDIDATES="${MAX_CANDIDATES:-4}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/nfs06/yangs/data/parquet_data/mt_distill_data/tower_zhen.oss.flash_gpe.${PROMPT_TYPE}.max${MAX_CANDIDATES}.parquet}"

.venv/bin/python -m data.run_oss_flash_gpe_sft_data \
  --data_path "${DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --prompt_type "${PROMPT_TYPE}" \
  --max_candidates "${MAX_CANDIDATES}"
