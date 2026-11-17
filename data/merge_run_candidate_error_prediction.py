"""Merge recursive Parquet inputs into one SFT candidate-error prediction call."""
from data.candidate_error_workflow import predict_directory
from inference.run_candidate_error_prediction import predict
from utils.hard_protocol import normalize_bool
from utils.helpers import validate_think_tag


def main(input_dir, output_dir, model_path, candidate_model='provided',
         scoring='score10', base_url=None, workers=8, temperature=0.1, top_p=1.0,
         max_tokens=8192, retry=2, vllm_kwargs=None, prompt_variant='critical_micro_fit',
         enable_thinking=False, think_tag='think'):
    validate_think_tag(think_tag)
    enable_thinking = normalize_bool(enable_thinking)
    return predict_directory(
        input_dir, output_dir, model_path, predictor=predict,
        candidate_model=candidate_model, scoring=scoring, prompt_variant=prompt_variant,
        base_url=base_url, workers=workers, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, retry=retry, vllm_kwargs=vllm_kwargs,
        enable_thinking=enable_thinking, think_tag=think_tag)


if __name__ == '__main__':
    import fire
    fire.Fire(main, serialize=lambda result: None)
