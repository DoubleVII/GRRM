import json

from eval.hard_eval_core import DEFAULT_MODEL, evaluate, summarize
from utils.hard_protocol import load_dataset_artifacts, output_paths, parse_data_ids, save


def main(data_id, predictions_dir, golden_dir, output_dir, mode="direct", model_path=DEFAULT_MODEL,
         temperature=0.0, top_p=1.0, max_tokens=8192, retry=3,
         reasoning_effort="medium", vllm_kwargs=None):
    data_ids = parse_data_ids(data_id)
    paths = output_paths(data_ids, output_dir, "evaluation", mode)
    predictions, prediction_configs = load_dataset_artifacts(data_ids, predictions_dir, "predictions", mode)
    golden, golden_configs = load_dataset_artifacts(data_ids, golden_dir, "golden")
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                   retry=retry, reasoning_effort=reasoning_effort, vllm_kwargs=vllm_kwargs)
    items, summary = evaluate(predictions, golden, model_path, **options)
    for did, path in paths.items():
        selected = [item for item in items if item["data_id"] == did]
        save(path, "evaluation", dict(model_path=model_path, data_id=did, mode=mode, **options,
             prediction_config=prediction_configs[did], golden_config=golden_configs[did]),
             selected, summary=dict(overall=summarize(selected),
                                    by_dataset={did: summary["by_dataset"][did]}))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
