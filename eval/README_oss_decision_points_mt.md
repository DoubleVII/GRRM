# GPT-OSS selective decision-points MT evaluation

This prompt-isolated variant replaces exhaustive source segmentation with a
small set of high-impact translation decision points. Stage 1 may return zero
to `MAX_DECISION_POINTS` local decisions and records whole-text semantic
constraints separately. Stage 2 drafts the complete translation directly from
the source before consulting those decisions, then performs a global fidelity
audit and polish pass.

The existing `json` and `codeblock` variants are unchanged. Candidate
confidence is intentionally unsupported for this prompt type.

Run all configured datasets:

```bash
CUDA_VISIBLE_DEVICES=0,3 ./run_oss_decision_points_mt_eval.sh
```

Small SeedX smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,3 \
DATA_IDS=seedx_challenge_zhen,seedx_challenge_enzh \
MAX_SAMPLES=8 \
RUNS=1 \
OUTPUT_PATH=results/oss_diverse_mt_eval.decision_points.polish.no-confidence.dp4.smoke.json \
./run_oss_decision_points_mt_eval.sh
```

The default result name is
`results/oss_diverse_mt_eval.decision_points.polish.no-confidence.dp4.json`.
