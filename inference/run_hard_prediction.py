"""Direct source difficulty prediction for SFT models, using the inst backend.

Examples (run from the repository root)::

    .venv/bin/python -m inference.run_hard_prediction \
        --input_file=/path/to/source.parquet --output_dir=/path/to/predictions \
        --model_path=/path/to/sft-model

    .venv/bin/python -m inference.run_hard_prediction \
        --data_id=ltb_v1_dev_scarcity_high --annotations_dir=/path/to/annotations \
        --output_dir=/path/to/predictions --model_path=/path/to/sft-model
"""
from data.run_oss_hard_annotation import _input_paths, load_items
from inference.hard_generation import generate
from inference.run_oss_hard_annotation import validate_annotation
from inference.sft_hard_prompts import SFT_HARD_PROMPT_VERSION, build_sft_hard_prompt
from utils.helpers import validate_think_tag
from utils.difficulty_fallback import resolve_prediction
from utils.difficulty_protocol import (
    indexed, load_annotations, prediction_sources, save_artifact, validate_runs,
)
from utils.hard_protocol import normalize_bool, output_paths, parse_data_ids


def predict(items, model_path, *, include_calibration=False, examples_per_score=2,
            max_span_chars=48, max_span_words=8, failure_mode="strict", runs: int = 1,
            enable_thinking=True, think_tag="think", **generation_options):
    """Generate run-major predictions from source text and language direction only."""
    validate_think_tag(think_tag)
    validate_runs(runs)
    if failure_mode not in {"strict", "fallback"}:
        raise ValueError("failure_mode must be strict or fallback")
    if any(type(limit) is not int or limit < 1 for limit in (max_span_chars, max_span_words)):
        raise ValueError("Span limits must be positive integers")
    include_calibration = normalize_bool(include_calibration)
    enable_thinking = normalize_bool(enable_thinking)
    build_sft_hard_prompt(dict(src_text="check", src_lang="en", trg_lang="zh"),
        include_calibration=include_calibration, examples_per_score=examples_per_score)
    clean = prediction_sources(items, "direct")
    repeated = [dict(item, run_id=run_id) for run_id in range(runs) for item in clean]
    prompts = [build_sft_hard_prompt(item, include_calibration=include_calibration,
        examples_per_score=examples_per_score) for item in repeated]
    limits = dict(max_span_chars=max_span_chars, max_span_words=max_span_words)
    results = generate(prompts,
        [lambda value, item=item: validate_annotation(value, item, **limits) for item in repeated],
        backend="inst", model_path=model_path, enable_thinking=enable_thinking,
        response_format="json", think_tag=think_tag, **generation_options)
    if len(results) != len(repeated):
        raise ValueError("Prediction output count mismatch")
    output = []
    for item, generation in zip(repeated, results):
        valid = generation.get("parsed") is not None
        prediction = dict(item, status="ok" if valid else "failed",
            result=generation.get("parsed"), generation=generation,
            error=None if valid else "Invalid difficulty prediction after retries")
        output.append(resolve_prediction(prediction, failure_mode=failure_mode, limits=limits))
    return output


def main(output_dir, model_path, data_id=None, annotations_dir=None, input_file=None,
         data_dir=None, max_samples: int = 0, temperature=0.9, top_p=1.0,
         max_tokens=8192, retry=3, enable_thinking=True, include_calibration=False,
         examples_per_score=2, max_span_chars=48, max_span_words=8,
         vllm_kwargs=None, failure_mode="strict", runs: int = 1, think_tag: str = "think"):
    """Predict direct difficulty from annotations or source Parquet; QE is unsupported."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("Provide an explicit model_path")
    validate_think_tag(think_tag)
    validate_runs(runs)
    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be a nonnegative integer")
    if annotations_dir is not None:
        if input_file is not None or data_dir is not None:
            raise ValueError("annotations_dir cannot be combined with input_file or data_dir")
        if data_id is None:
            raise ValueError("annotations_dir requires data_id")
        data_ids = parse_data_ids(data_id)
    else:
        data_ids = tuple(_input_paths(data_id, data_dir, input_file))
    paths = output_paths(data_ids, output_dir, "predictions", "direct")
    annotation_configs = {}
    if annotations_dir is not None:
        items, annotation_configs = load_annotations(data_ids, annotations_dir)
        if max_samples:
            items = [item for did in data_ids
                     for item in [x for x in items if x["data_id"] == did][:max_samples]]
    else:
        items = load_items(data_ids, data_dir, max_samples, input_file=input_file,
                           require_reference=False)
    indexed(items)
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        retry=retry, enable_thinking=normalize_bool(enable_thinking),
        include_calibration=normalize_bool(include_calibration), examples_per_score=examples_per_score,
        max_span_chars=max_span_chars, max_span_words=max_span_words,
        vllm_kwargs=vllm_kwargs, failure_mode=failure_mode, runs=runs, think_tag=think_tag)
    print(f"Prediction: backend=inst; mode=direct; sources={len(items)}; "
          f"runs={runs}; requests={len(items) * runs}", flush=True)
    output = predict(items, model_path, **options)
    config = dict(model_path=model_path, backend="inst", mode="direct",
        prompt_variant="sft", sft_prompt_version=SFT_HARD_PROMPT_VERSION,
        anchor_scores=[0, 2, 4, 6, 8, 10] if options["include_calibration"] else [],
        max_samples=max_samples, input_file=str(input_file) if input_file is not None else None,
        data_dir=str(data_dir) if data_dir is not None else None, **options)
    for did, path in paths.items():
        selected = [item for item in output if item["data_id"] == did]
        counts = {kind: sum(item["acceptance"] == kind for item in selected)
                  for kind in ("strict", "relaxed", "empty", "failed")}
        save_artifact(path, "predictions", dict(config, data_id=did,
            annotation_config=annotation_configs.get(did)), selected, summary=counts)
        print(f"Predictions saved: {path}; items={len(selected)}; acceptance={counts}", flush=True)
    return output


if __name__ == "__main__":
    import fire
    fire.Fire(main, serialize=lambda result: None)
