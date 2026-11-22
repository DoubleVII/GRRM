"""Merge recursive Parquet inputs into one INST candidate-error prediction call."""
from data.candidate_error_workflow import predict_directory
from inference.run_inst_candidate_error_prediction import predict
from utils.hard_protocol import normalize_bool


def main(input_dir, output_dir, model_path, candidate_model='provided',
         scoring='probability', base_url=None, workers=8, temperature=0.6,
         max_tokens=8192, retry=2, vllm_kwargs=None, prompt_variant='auto',
         top_p=1.0, enable_thinking=False):
    enable_thinking = normalize_bool(enable_thinking)
    return predict_directory(
        input_dir, output_dir, model_path, predictor=predict,
        candidate_model=candidate_model, scoring=scoring, prompt_variant=prompt_variant,
        base_url=base_url, workers=workers, temperature=temperature,
        max_tokens=max_tokens, retry=retry, vllm_kwargs=vllm_kwargs,
        top_p=top_p, enable_thinking=enable_thinking)


if __name__ == '__main__':
    import fire
    fire.Fire(main, serialize=lambda result: None)
