# Qwen SFT MT Evaluation

The trained models use a plain-text assistant protocol:

```text
<thinking>
reasoning
</thinking>
<response>
task-specific response
</response>
```

The outer protocol is parsed first. The response is then validated with the
existing direct translation, group post-edit, or SCD JSON/final-translation
parser. Invalid output is retried and never passed to the evaluator.
Task prompts intentionally omit output-format instructions and detailed
rubrics; the supervised responses teach the protocol and task preferences.

## SFT data

```bash
.venv/bin/python -m scripts.prepare_SFT_GPE_training_data \
  --data_path /home/nfs06/yangs/data/parquet_data/mt_distill_data/tower_zhen.oss.gpe.parquet \
  --output_path /home/nfs06/yangs/data/parquet_data/training_data/tower_zhen.oss.gpe.sft.parquet

.venv/bin/python -m scripts.prepare_SFT_SCD_training_data \
  --data_path /home/nfs06/yangs/data/parquet_data/mt_distill_data/tower_zhen.oss.scd.parquet \
  --output_path /home/nfs06/yangs/data/parquet_data/training_data/tower_zhen.oss.scd.sft.parquet
```

GPE produces four direct-MT examples and one group-post-edit example per source.
The command above writes them separately as
`tower_zhen.oss.gpe.sft.direct_mt.parquet` and
`tower_zhen.oss.gpe.sft.group_post_edit.parquet`, so their training mixture can
be configured independently. Use `--direct_output_path` and
`--post_edit_output_path` to override either derived path.
SCD produces one four-message conversation per source.

## Inference only

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m inference.run_qwen_sft_mt \
  --method scd \
  --model_path CHECKPOINT \
  --input_path INPUT.parquet \
  --output_path results/scd.inference.json
```

## OSS evaluation

```bash
CUDA_VISIBLE_DEVICES=0,3 METHOD=scd MODEL_PATH=CHECKPOINT \
  MODEL_LABEL=scd-sft bash run_qwen_sft_mt_eval.sh
```

To evaluate both trained methods sequentially:

```bash
CUDA_DEVICES=0,3 \
SCD_MODEL_PATH=SCD_CHECKPOINT \
GPE_MODEL_PATH=GPE_CHECKPOINT \
bash run_qwen_sft_mt_eval_all.sh
```

Qwen is released before the gpt-oss-120b reference-aware evaluator is loaded.
Set `RUN_GPE_DIRECT=true` to additionally evaluate the direct task of the GPE
checkpoint.
