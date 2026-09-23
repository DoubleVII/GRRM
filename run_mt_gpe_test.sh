set -e
export WANDB_DIR=/home/yangs/wandb_logs
export WANDB_MODE=offline

MT_MODEL_PATH=/home/nfs06/yangs/ckpt/Qwen/Qwen2.5-3B/verl/mt/v5
GPE_MODEL_PATH=/home/nfs06/yangs/ckpt/Qwen/Qwen2.5-3B/verl/pe/v3
MODEL_NAME=qwen2.5-3b.verl.mt.v5+pe.v3.wmt.runs1
RUNS=1
# DATA_IDS=seedx_challenge_zhen,seedx_challenge_enzh,wmt23_de_en,wmt23_ja_en,wmt23_ru_en,wmt23_uk_en,wmt23_zh_en,wmt24pp_en_de,wmt24pp_en_es,wmt24pp_en_fr,wmt24pp_en_it,wmt24pp_en_nl,wmt24pp_en_pt,wmt24pp_en_ja,wmt24pp_en_ko,wmt24pp_en_ru,wmt24pp_en_uk,wmt24pp_en_zh
# DATA_IDS=seedx_challenge_zhen,seedx_challenge_enzh
DATA_IDS=wmt23_de_en,wmt23_ja_en,wmt23_ru_en,wmt23_uk_en,wmt23_zh_en,wmt24pp_en_de,wmt24pp_en_es,wmt24pp_en_fr,wmt24pp_en_it,wmt24pp_en_nl,wmt24pp_en_pt,wmt24pp_en_ja,wmt24pp_en_ko,wmt24pp_en_ru,wmt24pp_en_uk,wmt24pp_en_zh

CUDA_VISIBLE_DEVICES=6,7 nohup python -m eval.run_mt_gpe_eval \
    --data_id $DATA_IDS \
    --model_path $MT_MODEL_PATH \
    --gpe_model_path $GPE_MODEL_PATH \
    --model_name $MODEL_NAME \
    --data_dir /home/nfs06/yangs/data/parquet_data/test_data_mt_with_notes \
    --sampling_n 4 \
    --temperature 0.6 \
    --top_p 0.9 \
    --max_new_tokens 8192 \
    --metrics '["bleurt","oss"]' \
    --prompt_type codeblock-think \
    --runs $RUNS \
    --bleurt_model_path /home/nfs06/yangs/metric_ckpt/BLEURT-20 \
    --oss_model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b > logs/mt_gpe_eval_${MODEL_NAME}.log 2>&1
