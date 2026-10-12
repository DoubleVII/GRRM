import json
from pathlib import Path

from eval.run_oss_diverse_mt_eval import (
    _load_data, _mean, _parse_data_ids, _score_translations, _summary_for_indices,
)
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_key_points_mt import pipeline_item, run_pipeline


def key_points_stats(results: list) -> dict:
    valid = [value for value in results if value is not None]
    point_counts = [len(value["key_points"]) for value in valid]
    candidate_counts = [len(point["translations"])
                        for value in valid for point in value["key_points"]]
    return {
        "valid_stage1": len(valid),
        "failed_stage1": len(results) - len(valid),
        "items_without_key_points": sum(count == 0 for count in point_counts),
        "mean_key_points_per_item": _mean(point_counts),
        "mean_candidates_per_key_point": _mean(candidate_counts),
    }


def main(
    data_id="seedx_challenge_zhen",
    model_path: str = "openai/gpt-oss-120b",
    output_path: str = "results/oss_key_points_mt_eval.json",
    max_samples: int = 16,
    seed: int = 42,
    reasoning_effort: str = "medium",
    key_points_temperature: float = 1.0,
    key_points_top_p: float = 1.0,
    stage1_max_tokens: int = 8192,
    final_temperature: float = 1.0,
    final_top_p: float = 1.0,
    final_max_tokens: int = 8192,
    runs: int = 1,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Generate once, then evaluate the final translations with repeated SQM scoring."""
    settings = {key: value for key, value in locals().items()
                if key not in {"data_id", "model_path", "output_path"}}
    if runs < 1:
        raise ValueError("runs must be at least 1")
    data_ids = _parse_data_ids(data_id)
    frame = _load_data(data_ids, max_samples, seed)
    print(f"Loaded {len(frame)} items from {', '.join(data_ids)}")
    model = init_oss_model(model_path, gpu_memory_utilization=gpu_memory_utilization,
                           max_model_len=max_model_len) if len(frame) else None
    output = run_pipeline(
        frame["src_text"].tolist(), frame["src_lang"].tolist(), frame["trg_lang"].tolist(),
        model=model, model_path=model_path, reasoning_effort=reasoning_effort,
        key_points_temperature=key_points_temperature, key_points_top_p=key_points_top_p,
        stage1_max_tokens=stage1_max_tokens, final_temperature=final_temperature,
        final_top_p=final_top_p, final_max_tokens=final_max_tokens, retry=retry,
    )
    translations = output["convergent"]["translations"]
    evaluation = _score_translations(frame, translations, model, model_path, runs=runs)
    scores = evaluation["scores"]
    scores_by_run = evaluation["scores_by_run"]
    responses_by_run = evaluation["responses_by_run"]
    summaries = {}
    stats = {}
    groups = {current: frame.index[frame["data_id"] == current].tolist()
              for current in data_ids}
    groups["overall"] = list(range(len(frame)))
    for name, indices in groups.items():
        summaries[name] = _summary_for_indices(indices, scores, scores_by_run, translations)
        stats[name] = key_points_stats([output["key_points"]["key_points"][i] for i in indices])
    items = []
    for i, row in frame.iterrows():
        items.append({
            "index": i, "data_id": row["data_id"], "source_index": int(row["source_index"]),
            "src_text": row["src_text"], "ref_text": row["trg_text"],
            "src_lang": row["src_lang"], "trg_lang": row["trg_lang"],
            **pipeline_item(output, i),
            "two_stage_score": scores[i],
            "two_stage_scores": [run[i] for run in scores_by_run],
            "two_stage_evaluator_response": responses_by_run[0][i],
            "two_stage_evaluator_responses": [run[i] for run in responses_by_run],
        })
    payload = {"method": "key_points_mt", "model_path": model_path,
               "data_ids": data_ids, "settings": settings,
               "key_points_stats": stats, "summary": summaries, "items": items}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"key_points_stats": stats, "summary": summaries}, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
