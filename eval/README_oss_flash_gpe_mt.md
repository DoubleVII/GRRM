# GPT-OSS FlashGPE MT evaluation

FlashGPE generates multiple diverse translations in one model response, then
uses one post-editing response to produce the final translation. It is separate
from the original GPE method, which samples candidates independently.

Run the fixed-four variant on CUDA devices 0 and 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PROMPT_TYPE=fixed_4 \
MAX_CANDIDATES=4 \
./run_oss_flash_gpe_mt_eval.sh
```

Supported prompt types are `fixed_4`, `adaptive`, and `fixed_16`.
`max_candidates` is a ceiling for adaptive variants; `fixed_4` requires 4 and
`fixed_16` requires 16. Results include the parsed candidates, the joint raw
candidate response, final FlashGPE output, evaluator runs, and candidate-count
statistics.

Inference without evaluation:

```bash
.venv/bin/python -m inference.run_oss_flash_gpe_mt \
  --input_path INPUT.parquet \
  --output_path results/oss_flash_gpe.inference.json \
  --max_candidates 4 \
  --prompt_type fixed_4
```
