#!/usr/bin/env bash
set -euo pipefail

export PROMPT_TYPE=decision_points
export POLISH="${POLISH:-true}"
export CANDIDATE_CONFIDENCE=false
export MAX_DECISION_POINTS="${MAX_DECISION_POINTS:-4}"

bash run_oss_diverse_mt_eval.sh
