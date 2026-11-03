from inference.hard_generation import generate
from inference.hard_prompts import build_prompt
from utils.hard_protocol import (load_dataset_artifacts, normalize_bool, output_paths,
                                 parse_data_ids, result_item, save, valid_candidate, validate_spans)

def predict(items, mode, model_path, *, backend="inst", **generation_options):
    """Shared prediction protocol; select the instruct or Harmony/OSS adapter."""
    # Validate mode even on empty input.
    if mode not in {"direct", "qe"}:
        raise ValueError("mode must be direct or qe")
    eligible = [item for item in items if valid_candidate(item)]
    results = generate(
        [build_prompt(item, mode) for item in eligible],
        [lambda value, item=item: validate_spans(value, item) for item in eligible],
        backend=backend, model_path=model_path, **generation_options)
    iterator = iter(results)
    return [result_item(item, next(iterator)) if valid_candidate(item)
            else result_item(item, error="Candidate translation failed") for item in items]


def main(data_id, candidates_dir, output_dir, model_path, mode="direct", temperature=0.9,
         top_p=1.0, max_tokens=8192, retry=3, enable_thinking=False, vllm_kwargs=None):
    data_ids = parse_data_ids(data_id)
    paths = output_paths(data_ids, output_dir, "predictions", mode)
    items, candidate_configs = load_dataset_artifacts(data_ids, candidates_dir, "candidates")
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                   retry=retry, enable_thinking=normalize_bool(enable_thinking),
                   vllm_kwargs=vllm_kwargs)
    output = predict(items, mode, model_path, **options)
    for did, path in paths.items():
        save(path, "predictions", dict(model_path=model_path, data_id=did, mode=mode,
             candidate_config=candidate_configs[did], **options),
             [item for item in output if item["data_id"] == did])


if __name__ == "__main__":
    import fire
    fire.Fire(main)
