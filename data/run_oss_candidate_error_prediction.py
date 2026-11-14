"""OSS candidate-error prediction over Parquet or JSONL."""
from data.candidate_error_input import load_items
from data.candidate_error_workflow import predict


def main(input_file, output_path, model_path, data_id=None, candidate_model='provided',
         src_lang=None, trg_lang=None, max_samples=0, scoring='probability',
         base_url=None, workers=8, temperature=0.1, max_tokens=8192, retry=2,
         vllm_kwargs=None, prompt_variant='critical_combined_minor', reasoning_effort='medium'):
    items = load_items(input_file, data_id, candidate_model, False, src_lang, trg_lang, max_samples)
    return predict(items, output_path, model_path, backend='oss', scoring=scoring,
                   base_url=base_url, workers=workers, temperature=temperature,
                   max_tokens=max_tokens, retry=retry, vllm_kwargs=vllm_kwargs,
                   prompt_variant=prompt_variant, reasoning_effort=reasoning_effort)


if __name__ == '__main__':
    import fire
    fire.Fire(main, serialize=lambda result: None)
