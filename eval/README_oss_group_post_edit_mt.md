# GPT-OSS group post-editing MT evaluation

This pipeline samples multiple direct translations with gpt-oss-120b, then
asks the same model to compare, combine, and post-edit those candidates into one
final translation. The final output uses the same reference-aware OSS evaluator,
dataset loading, sampling seed, and repeated evaluation statistics as the direct
and divergent/convergent pipelines.

Run all configured datasets on CUDA devices 0 and 1:

```bash
./run_oss_group_post_edit_mt_eval.sh
```

Defaults:

- `SAMPLING_N=4`
- candidate sampling: `temperature=0.8`, `top_p=0.95`
- group post-editing: `temperature=0.3`, `top_p=0.8`
- evaluator: `RUNS=4`
- output: `results/oss_group_post_edit_mt_eval.n4.json`

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DATA_IDS=seedx_challenge_zhen,seedx_challenge_enzh \
MAX_SAMPLES=16 \
SAMPLING_N=4 \
RUNS=4 \
OUTPUT_PATH=results/oss_group_post_edit_mt_eval.n4.smoke.json \
./run_oss_group_post_edit_mt_eval.sh
```

`SAMPLING_N` must be between 2 and the number of configured candidate labels
(currently 8). Each result item preserves all sampled candidates, raw candidate
responses, the final post-edit response, and all evaluator runs.

Inference without evaluation is also available:

```bash
.venv/bin/python -m inference.run_oss_group_post_edit_mt \
  --input_path parquet_data/test_data_mt/seedx_challenge_zhen.parquet \
  --output_path results/oss_group_post_edit_mt_inference.json \
  --model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b \
  --max_samples 16
```
