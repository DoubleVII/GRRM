"""SFT candidate-error prediction over Parquet or flat candidate JSONL.

Run from the repository root::

    .venv/bin/python -m data.run_candidate_error_prediction \
        --input_file=/path/to/candidates.parquet --output_path=/path/to/predictions.jsonl \
        --model_path=/path/to/sft-model --think_tag=thinking
"""
from data.candidate_error_input import load_items
from inference.run_candidate_error_prediction import predict
from utils.jsonl import write_jsonl
from utils.hard_protocol import normalize_bool
from utils.helpers import validate_think_tag


def main(input_file, output_path, model_path, data_id=None, candidate_model='provided',
         src_lang=None, trg_lang=None, max_samples=0, scoring='score10',
         base_url=None, workers=8, temperature=0.6, top_p=1.0, max_tokens=8192, retry=2,
         vllm_kwargs=None, prompt_variant='critical_micro_fit', enable_thinking=False,
         think_tag='think'):
    validate_think_tag(think_tag)
    enable_thinking = normalize_bool(enable_thinking)
    items = load_items(input_file, data_id, candidate_model, False, src_lang, trg_lang, max_samples)
    rows = predict(
        items, model_path, scoring=scoring, prompt_variant=prompt_variant,
        base_url=base_url, workers=workers, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, retry=retry, vllm_kwargs=vllm_kwargs,
        enable_thinking=enable_thinking, think_tag=think_tag)
    write_jsonl(output_path, rows)
    return rows


if __name__ == '__main__':
    import fire
    fire.Fire(main, serialize=lambda result: None)
