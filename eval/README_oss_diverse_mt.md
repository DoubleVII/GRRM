# GPT-OSS divergent/convergent MT evaluation

The diverse pipeline segments the source and generates local candidates, then compares and composes them into a final translation. Its output is scored by the existing reference-aware OSS SQM evaluator. The detailed JSON contains the source, reference, stage-1 output, final translation, raw final-channel response, evaluator response, and score.

Direct translation is intentionally kept in the separate `run_oss_direct_mt_eval.sh` pipeline described below.

Stage 1 supports two prompt modes through `PROMPT_TYPE` / `--prompt_type`:

- `json` (default): the original strict JSON schema, with structural validation and structural diversity statistics.
- `codeblock`: a prep-notes-style response containing brief analysis followed by one free-form Markdown code block. The complete response is passed to stage 2 without JSON parsing. Structural diversity statistics are `null` in this mode.

Two independent switches control the prompt ablations:

- `CANDIDATE_CONFIDENCE=true|false`: stage 1 labels every candidate `high`,
  `medium`, or `low`; stage 2 treats the label as a calibrated aid and still
  verifies the candidate against the source.
- `POLISH=true|false`: stage 2 performs an explicit whole-text polishing and
  source-verification pass after composing the candidates.

## Two-stage evaluation

The default command evaluates all configured datasets, enables both switches,
uses four evaluator runs, and uses CUDA devices 0 and 1:

```bash
./run_oss_diverse_mt_eval.sh
```

To run the relaxed mode on the same sample:

```bash
PROMPT_TYPE=codeblock \
./run_oss_diverse_mt_eval.sh
```

The default output path includes the prompt mode and both switch states, for
example `results/oss_diverse_mt_eval.codeblock.polish.confidence.json`, so
ablations do not overwrite each other.

Run the complete 2x2 ablation with commands such as:

```bash
POLISH=true  CANDIDATE_CONFIDENCE=true  ./run_oss_diverse_mt_eval.sh
POLISH=true  CANDIDATE_CONFIDENCE=false ./run_oss_diverse_mt_eval.sh
POLISH=false CANDIDATE_CONFIDENCE=true  ./run_oss_diverse_mt_eval.sh
POLISH=false CANDIDATE_CONFIDENCE=false ./run_oss_diverse_mt_eval.sh
```

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DATA_IDS=seedx_challenge_zhen,seedx_challenge_enzh \
MAX_SAMPLES=32 \
RUNS=4 \
OUTPUT_PATH=results/oss_diverse_mt_eval_32.json \
./run_oss_diverse_mt_eval.sh
```

`MAX_SAMPLES` applies independently to each dataset. `RUNS` repeats OSS scoring
of the same generated translations (the default launcher uses 4) without
rerunning translation inference. The summary reports the averaged score,
per-run dataset means, standard deviation, standard error, an approximate 95%
normal confidence interval, and failure counts. Each item retains all run scores
and evaluator responses. For JSON confidence runs, the diversity section also
reports the number and proportion of `high`, `medium`, and `low` candidates.

## Inference only

Use this two-stage entry point for an arbitrary parquet containing `src_text`, `src_lang`, and `trg_lang`:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m inference.run_oss_diverse_mt \
  --input_path parquet_data/test_data_mt/seedx_challenge_zhen.parquet \
  --output_path results/oss_diverse_mt_inference.json \
  --model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b \
  --max_samples 16
```

Set `max_samples=0` to process the complete input parquet.

## Direct-only baseline

To generate and evaluate only direct translations, without running either divergent or convergent stage:

```bash
./run_oss_direct_mt_eval.sh
```

It accepts the same `CUDA_VISIBLE_DEVICES`, `MODEL_PATH`, `DATA_IDS`,
`MAX_SAMPLES`, `OUTPUT_PATH`, and `RUNS` environment overrides as the
end-to-end script.
