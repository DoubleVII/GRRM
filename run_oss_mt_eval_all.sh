#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

run_direct() {
  OUTPUT_PATH=results/oss_direct_mt_eval.runs4.json \
  RUNS=4 \
  bash run_oss_direct_mt_eval.sh \
    2>&1 | tee -a logs/oss_direct_mt_eval.runs4.log
}

run_diverse() {
  local prompt_type="$1"
  local polish="$2"
  local confidence="$3"
  local polish_variant="no-polish"
  local confidence_variant="no-confidence"
  [[ "${polish}" == "true" ]] && polish_variant="polish"
  [[ "${confidence}" == "true" ]] && confidence_variant="confidence"
  local variant="${prompt_type}.${polish_variant}.${confidence_variant}"

  PROMPT_TYPE="${prompt_type}" \
  POLISH="${polish}" \
  CANDIDATE_CONFIDENCE="${confidence}" \
  RUNS=4 \
  OUTPUT_PATH="results/oss_diverse_mt_eval.${variant}.json" \
  bash run_oss_diverse_mt_eval.sh \
    2>&1 | tee -a "logs/oss_diverse_mt_eval.${variant}.log"
}

run_group_post_edit() {
  local sampling_n="${1:-4}"

  SAMPLING_N="${sampling_n}" \
  RUNS=4 \
  OUTPUT_PATH="results/oss_group_post_edit_mt_eval.n${sampling_n}.json" \
  bash run_oss_group_post_edit_mt_eval.sh \
    2>&1 | tee -a "logs/oss_group_post_edit_mt_eval.n${sampling_n}.log"
}

run_flash_gpe() {
  # prompt_type: fixed_4, fixed_16, or adaptive (configurable maximum).
  local prompt_type="${1:-fixed_4}"
  local max_candidates="${2:-4}"

  PROMPT_TYPE="${prompt_type}" \
  MAX_CANDIDATES="${max_candidates}" \
  RUNS=4 \
  OUTPUT_PATH="results/oss_flash_gpe_mt_eval.${prompt_type}.max${max_candidates}.json" \
  bash run_oss_flash_gpe_mt_eval.sh \
    2>&1 | tee -a "logs/oss_flash_gpe_mt_eval.${prompt_type}.max${max_candidates}.log"
}

# Comment out any line below to skip that experiment.
# run_direct
# run_group_post_edit 4
run_flash_gpe fixed_4 4
# run_flash_gpe fixed_16 16
# run_flash_gpe adaptive 16

# JSON 2x2: polish x candidate confidence
# run_diverse json false false
# run_diverse json true false
# run_diverse json false true
# run_diverse json true true

# Codeblock 2x2: polish x candidate confidence
# run_diverse codeblock false false
# run_diverse codeblock true false
# run_diverse codeblock false true
# run_diverse codeblock true true
