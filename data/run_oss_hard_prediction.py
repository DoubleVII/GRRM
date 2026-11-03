"""Collect source difficulty predictions directly from parquet using OSS."""
import json
from pathlib import Path

from data.run_oss_hard_annotation import _input_paths, load_items
from inference.difficulty_prompts import VARIANTS
from inference.run_oss_hard_prediction import predict
from utils.difficulty_protocol import prediction_sources, save_artifact
from utils.hard_protocol import check_output


def main(data_id=None, output_dir=None, model_path=None, data_dir=None, max_samples=0,
         temperature=0.9, top_p=1.0, max_tokens=8192, retry=3, reasoning_effort="medium",
         examples_per_score=2, max_span_chars=48, max_span_words=8, vllm_kwargs=None,
         input_file=None, prompt_variant="full", failure_mode="strict"):
    """Predict direct source difficulty once per row.

    Defaults to the full prompt and strict validation. No annotation
    artifacts or reference translations are required. Existing files are rejected.
    """
    if output_dir is None:
        raise ValueError("Provide output_dir")
    if model_path is None or not str(model_path).strip():
        raise ValueError("Provide an explicit model_path")
    if prompt_variant not in VARIANTS:
        raise ValueError(f"prompt_variant must be one of {VARIANTS}")
    if failure_mode not in {"strict", "fallback"}:
        raise ValueError("failure_mode must be strict or fallback")
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("reasoning_effort must be low, medium, or high")
    data_ids = tuple(_input_paths(data_id, data_dir, input_file))
    paths = {did: Path(output_dir) / f"{did}.direct.predictions.json" for did in data_ids}
    for path in paths.values():
        check_output(path)
    items = load_items(data_ids, data_dir, max_samples, input_file=input_file,
                       require_reference=False)
    items = prediction_sources(items, "direct")
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                   retry=retry, reasoning_effort=reasoning_effort,
                   examples_per_score=examples_per_score, max_span_chars=max_span_chars,
                   max_span_words=max_span_words, vllm_kwargs=vllm_kwargs,
                   prompt_variant=prompt_variant, failure_mode=failure_mode)
    output = predict(items, "direct", model_path, **options)
    summary = {}
    for did, path in paths.items():
        subset = [item for item in output if item["data_id"] == did]
        counts = dict(total=len(subset), valid=sum(item["status"] == "ok" for item in subset),
                      failed=sum(item["status"] == "failed" for item in subset))
        config = dict(model_path=model_path, backend="oss", mode="direct",
                      anchor_scores=[0, 6, 10] if prompt_variant == "compact3" else [0, 2, 4, 6, 8, 10], data_id=did,
                      data_dir=str(data_dir) if data_dir is not None else None,
                      max_samples=max_samples, **options)
        if input_file is not None:
            config["input_file"] = str(input_file)
        if prompt_variant in {"compact3", "compact6"}:
            config["examples_per_score"] = 1
        save_artifact(path, "predictions", config, subset, summary=counts)
        summary[did] = dict(path=str(path), **counts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
