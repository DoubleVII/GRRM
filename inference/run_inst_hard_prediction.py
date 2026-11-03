"""Predict intrinsic 0–10 source difficulty with an instruct model."""
from inference.hard_generation import generate
from utils.difficulty_fallback import resolve_prediction
from inference.difficulty_prompts import build_prediction_prompt, DEFAULT_INST_VARIANT
from inference.run_oss_hard_annotation import validate_annotation
from utils.difficulty_protocol import (load_annotations, prediction_sources, source_item,
                                       save_artifact, indexed, load_candidates, validate_runs)
from utils.hard_protocol import (normalize_bool, output_paths,
                                 parse_data_ids, valid_candidate)


def predict(items, mode, model_path, *, backend="inst", prompt_variant=None,
            examples_per_score=2, max_span_chars=48, max_span_words=8, failure_mode="strict", runs: int = 1, **generation_options):
    if mode not in {"direct", "qe"}:
        raise ValueError("mode must be direct or qe")
    if backend not in {"inst", "oss"}:
        raise ValueError("backend must be inst or oss")
    if failure_mode not in ("strict", "fallback"):
        raise ValueError("failure_mode must be strict or fallback")
    validate_runs(runs)
    indexed(items)
    variant = prompt_variant or (DEFAULT_INST_VARIANT if backend == "inst" else "full")
    clean = [source_item(item, mode) for item in items]
    # Run-major order: one flattened request list, one shared generation call.
    repeated = [dict(item, run_id=run_id) for run_id in range(runs) for item in clean]
    eligible = [item for item in repeated if mode == "direct" or valid_candidate(item)]
    prompts = [build_prediction_prompt(item, mode, variant=variant,
               examples_per_score=examples_per_score, max_span_chars=max_span_chars,
               max_span_words=max_span_words) for item in eligible]
    # Validate configuration even if every QE candidate is absent.
    if not eligible:
        build_prediction_prompt(dict(src_text="", src_lang="en", trg_lang="zh"), mode,
            variant=variant, examples_per_score=examples_per_score,
            max_span_chars=max_span_chars, max_span_words=max_span_words)
    results = generate(prompts,
        [lambda value, item=item: validate_annotation(value, item,
             max_span_chars=max_span_chars, max_span_words=max_span_words) for item in eligible],
        backend=backend, model_path=model_path, **generation_options)
    if len(results) != len(eligible):
        raise ValueError("Prediction output count mismatch")
    iterator = iter(results)
    output = []
    for item in repeated:
        result = next(iterator) if mode == "direct" or valid_candidate(item) else None
        valid = result is not None and result.get("parsed") is not None
        output.append(dict(item, status="ok" if valid else "failed",
            result=result.get("parsed") if result else None, generation=result,
            error=None if valid else ("Missing or failed candidate translation" if result is None
                                      else "Invalid difficulty prediction after retries")))
    return [resolve_prediction(x, failure_mode=failure_mode,
                limits=dict(max_span_chars=max_span_chars, max_span_words=max_span_words)) for x in output]


def run(data_id, annotations_dir, output_dir, model_path, *, backend="inst", mode="direct",
        candidates_dir=None, prompt_variant=None, examples_per_score=2,
        max_span_chars=48, max_span_words=8, failure_mode="strict", runs: int = 1, **options):
    validate_runs(runs)
    data_ids = parse_data_ids(data_id)
    paths = output_paths(data_ids, output_dir, "predictions", mode)
    annotations, annotation_configs = load_annotations(data_ids, annotations_dir)
    candidates, candidate_configs = None, None
    if mode == "qe":
        if candidates_dir is None:
            raise ValueError("QE requires candidates_dir")
        candidates, candidate_configs = load_candidates(data_ids, candidates_dir)
    items = prediction_sources(annotations, mode, candidates)
    variant = prompt_variant or (DEFAULT_INST_VARIANT if backend == "inst" else "full")
    config = dict(model_path=model_path, backend=backend, mode=mode, prompt_variant=variant, failure_mode=failure_mode, runs=runs,
                  examples_per_score=examples_per_score if variant == "full" else 1,
                  anchor_scores=[0, 6, 10] if variant == "compact3" else [0, 2, 4, 6, 8, 10],
                  max_span_chars=max_span_chars, max_span_words=max_span_words, **options)
    output = predict(items, mode, model_path, backend=backend, prompt_variant=variant,
        examples_per_score=examples_per_score, max_span_chars=max_span_chars,
        max_span_words=max_span_words, failure_mode=failure_mode, runs=runs, **options)
    for did, path in paths.items():
        save_artifact(path, "predictions", dict(config, data_id=did,
            annotation_config=annotation_configs[did],
            candidate_config=candidate_configs[did] if candidate_configs else None),
            [item for item in output if item["data_id"] == did])
    return output


def main(data_id, annotations_dir, output_dir, model_path, mode="direct", candidates_dir=None,
         temperature=0.9, top_p=1.0, max_tokens=8192, retry=3, enable_thinking=False,
         prompt_variant=None, examples_per_score=2, max_span_chars=48, max_span_words=8,
         vllm_kwargs=None, failure_mode="strict", runs: int = 1):
    return run(data_id, annotations_dir, output_dir, model_path, mode=mode, failure_mode=failure_mode, runs=runs,
        candidates_dir=candidates_dir, prompt_variant=prompt_variant,
        examples_per_score=examples_per_score, max_span_chars=max_span_chars,
        max_span_words=max_span_words, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, retry=retry, enable_thinking=normalize_bool(enable_thinking),
        vllm_kwargs=vllm_kwargs)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
