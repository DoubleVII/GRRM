# OSS MT SFT Data Collection

The segment-based method is named **Segment-Level Candidate Deliberation
(SCD)**. SCD first explores translation candidates for source segments and
then deliberates over them to produce and polish a complete translation.

Both collectors use the existing gpt-oss inference implementations and write
Parquet data. Rows are saved only when every assistant response can be parsed,
the parsed value matches the inference result, and every turn has non-empty
thinking content. Thinking and final-channel response text are always stored
in separate columns.

## SCD

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m data.run_oss_scd_sft_data \
  --data_path INPUT.parquet \
  --output_path OUTPUT.scd.parquet \
  --model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b
```

The default is the JSON SCD prompt with whole-translation polishing enabled.
The two supervision stages are stored as:

- `scd_stage1_prompt`, `scd_stage1_thinking`, `scd_stage1_response`,
  `scd_stage1_parsed`
- `scd_stage2_prompt`, `scd_stage2_thinking`, `scd_stage2_response`,
  `scd_translation`

Use `--prompt_type codeblock` to collect the looser Stage 1 variant. JSON is
recommended for the initial SFT experiment because its candidate structure is
validated field by field.

## Group Post-Editing

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m data.run_oss_gpe_sft_data \
  --data_path INPUT.parquet \
  --output_path OUTPUT.gpe.parquet \
  --model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b \
  --sampling_n 4
```

Stage 1 contains `sampling_n` independent direct-translation completions.
FlashGPE is a separate collection method:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m data.run_oss_flash_gpe_sft_data \
  --data_path INPUT.parquet \
  --output_path OUTPUT.flash_gpe.fixed_4.parquet \
  --model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b \
  --prompt_type markdown \
  --min_candidates 2 \
  --max_candidates 8
```

FlashGPE Stage 1 uses numbered Markdown headings (`# Candidate 1`, etc.) and
stores one prompt, thinking trace, and response whose parsed candidates may be
multiline. Stage 2 uses `# Step-by-step Analysis` and `# Final Translation`.
The deprecated JSON protocol is retained under `inference/legacy/` only for
reproducing historical rows. Its columns are:

- `flash_gpe_stage1_prompt`, `flash_gpe_stage1_thinking`,
  `flash_gpe_stage1_response`, `flash_gpe_candidates`
- `flash_gpe_stage2_prompt`, `flash_gpe_stage2_thinking`,
  `flash_gpe_stage2_response`, `flash_gpe_translation`
- `flash_gpe_prompt_type`, `flash_gpe_max_candidates`,
  `flash_gpe_target_candidate_count`, `flash_gpe_candidate_count`,
  `flash_gpe_protocol`, `flash_gpe_parser_valid`

The FlashGPE SFT preparation script writes `flash_gpe_candidates` and
`flash_gpe` datasets. It also accepts legacy rows marked with
`gpe_candidate_generation=single_call`; independent GPE rows remain owned by
`prepare_SFT_GPE_training_data.py`.

Both scripts preserve all columns from the input rows. Required input columns
default to `src_text`, `src_lang`, and `trg_lang`; their names are configurable.
