"""Merge recursive Parquet inputs into one INST candidate-error prediction call."""
from data.candidate_error_workflow import predict_directory
from inference.run_inst_candidate_error_prediction import predict


def main(input_dir, output_dir, model_path, candidate_model='provided',
         scoring='probability', base_url=None, workers=8, temperature=0.1,
         max_tokens=8192, retry=2, vllm_kwargs=None, prompt_variant='auto',
         reasoning_effort='medium'):
    return predict_directory(
        input_dir, output_dir, model_path, predictor=predict,
        candidate_model=candidate_model, scoring=scoring, prompt_variant=prompt_variant,
        base_url=base_url, workers=workers, temperature=temperature,
        max_tokens=max_tokens, retry=retry, vllm_kwargs=vllm_kwargs,
        reasoning_effort=reasoning_effort)


if __name__ == '__main__':
    import fire
    fire.Fire(main, serialize=lambda result: None)
