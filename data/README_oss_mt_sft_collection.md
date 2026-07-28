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

Stage 1 contains `sampling_n` independent ordinary direct-translation
completions. Stage 2 receives those complete translations and performs group
post-editing with the same prompt used by the OSS GPE evaluation pipeline. The
columns are:

- `gpe_stage1_prompts`, `gpe_stage1_thinking`, `gpe_stage1_responses`,
  `gpe_stage1_translations`
- `gpe_stage2_prompt`, `gpe_stage2_thinking`, `gpe_stage2_response`,
  `gpe_translation`

Both scripts preserve all columns from the input rows. Required input columns
default to `src_text`, `src_lang`, and `trg_lang`; their names are configurable.
