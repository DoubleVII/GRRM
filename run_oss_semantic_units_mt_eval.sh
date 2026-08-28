#!/usr/bin/env bash
set -euo pipefail

export PROMPT_TYPE=semantic_units
export POLISH="${POLISH:-true}"
export CANDIDATE_CONFIDENCE="${CANDIDATE_CONFIDENCE:-false}"

bash run_oss_diverse_mt_eval.sh
