
export WANDB_DIR=/home/yangs/wandb_logs
export WANDB_MODE=offline

DATA_IDS=seedx_challenge_zhen,seedx_challenge_enzh,wmt23_de_en,wmt23_ja_en,wmt23_ru_en,wmt23_uk_en,wmt23_zh_en,wmt24pp_en_de,wmt24pp_en_es,wmt24pp_en_fr,wmt24pp_en_it,wmt24pp_en_nl,wmt24pp_en_pt,wmt24pp_en_ja,wmt24pp_en_ko,wmt24pp_en_ru,wmt24pp_en_uk,wmt24pp_en_zh
MT_MODEL_PATH=/home/nfs06/yangs/ckpt/Qwen/Qwen2.5-3B/verl/fused_flash_gpe/v1
MODEL_NAME=qwen2.5-3b.verl.ffgpe.v1

DATA_IDS=seedx_challenge_zhen,seedx_challenge_enzh,wmt23_zh_en,wmt24pp_en_zh

python -m eval.run_mt_ffgpe_eval_legacy \
    --data_id $DATA_IDS \
    --model_path $MT_MODEL_PATH \
    --model_name $MODEL_NAME \
    --temperature 0.6 \
    --top_p 0.9 \
    --max_new_tokens 8192 \
    --metrics '["bleurt","oss"]' \
    --prompt_type fixed_4 \
    --runs 4 \
    --bleurt_model_path /home/nfs06/yangs/metric_ckpt/BLEURT-20 \
    --oss_model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b > logs/mt_eval_${MODEL_NAME}.zhen.log 2>&1




python -m eval.run_mt_ffgpe_eval_legacy \
    --data_id $DATA_IDS \
    --model_path $MT_MODEL_PATH \
    --model_name $MODEL_NAME \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_new_tokens 8192 \
    --metrics '["bleurt","oss"]' \
    --prompt_type fixed_4 \
    --runs 4 \
    --bleurt_model_path /home/nfs06/yangs/metric_ckpt/BLEURT-20 \
    --oss_model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b > logs/mt_eval_${MODEL_NAME}.zhen.temp8.log 2>&1



python -m eval.run_mt_ffgpe_eval_legacy \
    --data_id $DATA_IDS \
    --model_path $MT_MODEL_PATH \
    --model_name $MODEL_NAME \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_new_tokens 8192 \
    --metrics '["bleurt","oss"]' \
    --prompt_type fixed_4 \
    --runs 4 \
    --bleurt_model_path /home/nfs06/yangs/metric_ckpt/BLEURT-20 \
    --oss_model_path /home/zfs01/yangs/LLM/openai/gpt-oss-120b > logs/mt_eval_${MODEL_NAME}.zhen.temp8topp95.log 2>&1
