"""Predict intrinsic 0–10 difficulty using the Harmony/OSS adapter."""
from inference.run_inst_hard_prediction import predict as _predict, run


def predict(items, mode, model_path, *, runs: int = 1, **generation_options):
    return _predict(items, mode, model_path, backend="oss", runs=runs, **generation_options)


def main(data_id, annotations_dir, output_dir, model_path, mode="direct",
         candidates_dir=None, temperature=0.9, top_p=1.0, max_tokens=8192, retry=3,
         reasoning_effort="medium", prompt_variant="full", examples_per_score=2,
         max_span_chars=48, max_span_words=8, vllm_kwargs=None, failure_mode="strict", runs: int = 1):
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("reasoning_effort must be low, medium, or high")
    return run(data_id, annotations_dir, output_dir, model_path, backend="oss", mode=mode, failure_mode=failure_mode, runs=runs,
        candidates_dir=candidates_dir, prompt_variant=prompt_variant,
        examples_per_score=examples_per_score, max_span_chars=max_span_chars,
        max_span_words=max_span_words, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, retry=retry, reasoning_effort=reasoning_effort,
        vllm_kwargs=vllm_kwargs)


if __name__ == "__main__":
    import fire
    # Keep the Python return value without dumping full artifacts into CLI logs.
    fire.Fire(main, serialize=lambda result: None)
