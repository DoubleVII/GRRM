#!/usr/bin/env bash
set -euo pipefail

: "${SCD_MODEL_PATH:?Set SCD_MODEL_PATH to the trained SCD checkpoint}"
: "${GPE_MODEL_PATH:?Set GPE_MODEL_PATH to the trained GPE checkpoint}"
: "${FLASH_GPE_MODEL_PATH:?Set FLASH_GPE_MODEL_PATH to the trained FlashGPE checkpoint}"

CUDA_DEVICES="${CUDA_DEVICES:-0,3}"
RUNS="${RUNS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DATA_IDS="${DATA_IDS:-seedx_challenge_zhen,seedx_challenge_enzh,wmt23_zh_en,wmt24pp_en_zh}"
RUN_GPE_DIRECT="${RUN_GPE_DIRECT:-false}"
MAX_CANDIDATES="${MAX_CANDIDATES:-4}"
PROMPT_TYPE="${PROMPT_TYPE:-fixed_4}"
mkdir -p logs results

run_one() {
  local method="$1"
  local model_path="$2"
  local label="$3"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  METHOD="${method}" MODEL_PATH="${model_path}" MODEL_LABEL="${label}" \
  RUNS="${RUNS}" MAX_SAMPLES="${MAX_SAMPLES}" DATA_IDS="${DATA_IDS}" \
  MAX_CANDIDATES="${MAX_CANDIDATES}" PROMPT_TYPE="${PROMPT_TYPE}" \
  bash run_qwen_sft_mt_eval.sh 2>&1 | tee "logs/qwen_sft_mt_eval.${label}.${method}.log"
}

run_one scd "${SCD_MODEL_PATH}" scd-sft
run_one group_post_edit "${GPE_MODEL_PATH}" gpe-sft
run_one flash_gpe "${FLASH_GPE_MODEL_PATH}" flash-gpe-sft
if [[ "${RUN_GPE_DIRECT}" == "true" ]]; then
  run_one direct "${GPE_MODEL_PATH}" gpe-sft
fi
