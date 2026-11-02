# Reference-assisted OSS difficulty annotation

Run from the repository root with the development environment. The model receives
the source, translation direction, reference translation,
and optional expert `comment`. Candidate translations are not used. References
and comments help identify intended meanings and translation difficulties; the
model still scores source difficulty and quotes spans only from the source.
The default model is `openai/gpt-oss-120b`; a local checkpoint can be supplied with
`--model_path`. Generation uses the existing local vLLM/Harmony OSS adapter.

```bash
.venv/bin/python -m data.run_oss_hard_annotation \
  --data_id=wmt24_enzh,seedx_challenge,ltb_v1_eval_scarcity_high \
  --data_dir=/home/nfs06/yangs/data/parquet_data/test_data_mt \
  --output_dir=results/oss_hard_annotations \
  --max_samples=10 \
  --reasoning_effort=medium \
  --vllm_kwargs='{"tensor_parallel_size": 4}'
```

`data_id` accepts one or several IDs from `utils/config.py`. Without `data_dir`,
configured paths are used. With `data_dir`, the loader first looks for
`<data_id>.parquet`, then for the configured path relative to `test_data_mt`.
Custom IDs require `<data_dir>/<data_id>.parquet`. Required fields are `src_text`,
`src_lang`, `trg_lang`, and `trg_text` (reference translation); fixed language
directions can also come from the config.
`max_samples=0` selects all rows; other values select that many rows per dataset.
References are passed to the model and saved as `ref_text`. The `comment` column
is optional; missing, null, and blank comments are omitted from model context.
Row IDs are zero-based parquet positions. LTB IDs are retained when present.

Annotate a single text with its reference translation:

```bash
.venv/bin/python -m inference.run_oss_hard_annotation \
  --src_text='我的老板除了CPU我什么也不会。' --src_lang=zh --trg_lang=en \
  --ref_text='My boss only knows how to gaslight me.' \
  --comment='CPU is slang for workplace psychological manipulation here.' \
  --vllm_kwargs='{"tensor_parallel_size": 4}'
```

Each dataset produces `<data_id>.oss.difficulty_annotations.json`, with the rubric
version, generation configuration, success/failure counts, and per-row results.
Existing output files are never overwritten; all input datasets and output paths
are checked before generation. Results contain an integer text score (0–10),
English analysis, and exact source spans with integer scores (1–10) and English
explanations. The text score equals the highest span score; 0 requires no spans.
Raw responses and failed rows are preserved. Invalid responses are retried up to
`retry` times after the first attempt (default 3).

Prompts prefer short lookup expressions (1–5 words or 2–12 characters for scripts
without word spaces). Additional hard guards default to 48 characters and 8
whitespace-separated words per span, with no line breaks; adjust via
`--max_span_chars` and `--max_span_words` when necessary. These are upper bounds,
not recommended lengths. Quotes must match the source exactly and duplicates are
rejected. `--examples_per_score` controls the number of direction-aware anchors.

These artifacts use the independent `difficulty_annotations` stage. The old
candidate-error golden files and their evaluation scripts use a different
contract and cannot directly consume them. Calibration examples are drawn from
the test datasets; exclude those rows and duplicate LTB IDs from held-out
calibration evaluation (provenance is in `DIFFICULTY_EXAMPLES`).

Prompt version `reference-assisted-difficulty-v2` distinguishes these annotations
from the earlier source-only annotations. Single-text CLI calls require
`--ref_text`; `--comment` is optional.
