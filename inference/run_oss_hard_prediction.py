"""Predict translation difficulties with OSS, using the shared direct/QE protocol."""
from eval.hard_eval_core import DEFAULT_MODEL
from inference.run_inst_hard_prediction import predict as _predict
from utils.hard_protocol import (load_dataset_artifacts, output_paths,
                                 parse_data_ids, save)


def predict(items, mode, model_path=DEFAULT_MODEL, **generation_options):
    return _predict(items, mode, model_path, backend="oss", **generation_options)


def main(data_id, candidates_dir, output_dir, model_path=DEFAULT_MODEL, mode="direct",
         temperature=0.9, top_p=1.0, max_tokens=8192, retry=3,
         reasoning_effort="medium", vllm_kwargs=None):
    data_ids = parse_data_ids(data_id)
    paths = output_paths(data_ids, output_dir, "predictions", mode)
    items, candidate_configs = load_dataset_artifacts(data_ids, candidates_dir, "candidates")
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                   retry=retry, reasoning_effort=reasoning_effort, vllm_kwargs=vllm_kwargs)
    output = predict(items, mode, model_path, **options)
    for did, path in paths.items():
        save(path, "predictions", dict(model_path=model_path, data_id=did, mode=mode,
             candidate_config=candidate_configs[did], **options),
             [item for item in output if item["data_id"] == did])


if __name__ == "__main__":
    import fire
    fire.Fire(main)
