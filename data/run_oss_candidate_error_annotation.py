"""Reference-assisted OSS silver annotation over Parquet or JSONL."""
from data.candidate_error_input import load_items
from data.candidate_error_workflow import annotate


def main(input_file, output_dir, model_path, data_id=None, candidate_model='provided',
         src_lang=None, trg_lang=None, max_samples=0, review=True, base_url=None,
         workers=8, temperature=0.1, max_tokens=4096, retry=2, vllm_kwargs=None,
         prompt_variant='legacy', workflow='review', samples=3, min_successful_samples=2,
         reasoning_effort='medium'):
    items = load_items(input_file, data_id, candidate_model, True, src_lang, trg_lang, max_samples)
    return annotate(items, output_dir, model_path, review=review, base_url=base_url,
                    workers=workers, temperature=temperature, max_tokens=max_tokens,
                    retry=retry, vllm_kwargs=vllm_kwargs, prompt_variant=prompt_variant,
                    workflow=workflow, samples=samples, min_successful_samples=min_successful_samples,
                    reasoning_effort=reasoning_effort)


if __name__ == '__main__':
    import fire
    fire.Fire(main, serialize=lambda result: None)
