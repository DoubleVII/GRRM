import json
from pathlib import Path
from typing import Optional

from eval.run_oss_diverse_mt_eval import (
    _load_data,
    _parse_data_ids,
    _score_translations,
    _summary_for_indices,
)
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_group_post_edit_mt import (
    run_pipeline,
    validate_sampling_n,
)


def main(
    data_id="seedx_challenge_zhen",
    model_path: str = "/home/zfs01/yangs/LLM/openai/gpt-oss-120b",
    output_path: Optional[str] = None,
    max_samples: int = 16,
    seed: int = 42,
    sampling_n: int = 4,
    reasoning_effort: str = "medium",
    sampling_temperature: float = 0.8,
    sampling_top_p: float = 0.95,
    sampling_max_tokens: int = 4096,
    post_edit_temperature: float = 0.3,
    post_edit_top_p: float = 0.8,
    post_edit_max_tokens: int = 4096,
    retry: int = 3,
    runs: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Generate, group-post-edit, and evaluate gpt-oss translations."""
    validate_sampling_n(sampling_n)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if output_path is None:
        output_path = f"results/oss_group_post_edit_mt_eval.n{sampling_n}.json"
    data_ids = _parse_data_ids(data_id)
    frame = _load_data(data_ids, max_samples, seed)
    print(f"Loaded {len(frame)} items from {', '.join(data_ids)}")

    model = init_oss_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    pipeline = run_pipeline(
        frame["src_text"].tolist(),
        frame["src_lang"].tolist(),
        frame["trg_lang"].tolist(),
        model=model,
        model_path=model_path,
        sampling_n=sampling_n,
        reasoning_effort=reasoning_effort,
        sampling_temperature=sampling_temperature,
        sampling_top_p=sampling_top_p,
        sampling_max_tokens=sampling_max_tokens,
        post_edit_temperature=post_edit_temperature,
        post_edit_top_p=post_edit_top_p,
        post_edit_max_tokens=post_edit_max_tokens,
        retry=retry,
    )
    translations = pipeline["post_edit"]["translations"]
    evaluation = _score_translations(
        frame, translations, model, model_path, runs=runs
    )
    scores = evaluation["scores"]
    scores_by_run = evaluation["scores_by_run"]
    evaluator_responses_by_run = evaluation["responses_by_run"]

    summaries = {}
    for current_data_id in (*data_ids, "overall"):
        indices = (
            list(range(len(frame)))
            if current_data_id == "overall"
            else frame.index[frame["data_id"] == current_data_id].tolist()
        )
        summary = _summary_for_indices(
            indices,
            scores,
            scores_by_run,
            translations,
            score_name="group_post_edit_mean",
        )
        summary["candidate_generation_failures"] = sum(
            sampling_n - pipeline["usable_candidate_counts"][index]
            for index in indices
        )
        candidate_counts = [
            pipeline["usable_candidate_counts"][index] for index in indices
        ]
        summary["candidate_count_mean"] = (
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else None
        )
        summary["candidate_count_min"] = min(candidate_counts, default=None)
        summary["candidate_count_max"] = max(candidate_counts, default=None)
        summary["candidate_count_distribution"] = {
            str(count): candidate_counts.count(count)
            for count in sorted(set(candidate_counts))
        }
        summaries[current_data_id] = summary

    items = []
    for index, row in frame.iterrows():
        items.append({
            "index": index,
            "data_id": row["data_id"],
            "source_index": int(row["source_index"]),
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "src_text": row["src_text"],
            "ref_text": row["trg_text"],
            "mt_candidates": pipeline["sampling"]["translations"][index],
            "candidate_responses": pipeline["sampling"]["responses"][index],
            "candidate_thinking": pipeline["sampling"]["thinking"][index],
            "usable_candidate_count": pipeline["usable_candidate_counts"][index],
            "group_post_edit_translation": translations[index],
            "group_post_edit_response": pipeline["post_edit"]["responses"][index],
            "group_post_edit_thinking": pipeline["post_edit"]["thinking"][index],
            "group_post_edit_score": scores[index],
            "group_post_edit_evaluator_response": evaluator_responses_by_run[0][
                index
            ],
            "group_post_edit_scores": [
                run_scores[index] for run_scores in scores_by_run
            ],
            "group_post_edit_evaluator_responses": [
                run_responses[index]
                for run_responses in evaluator_responses_by_run
            ],
        })

    payload = {
        "model_path": model_path,
        "data_ids": data_ids,
        "max_samples_per_dataset": max_samples,
        "seed": seed,
        "settings": {
            "sampling_n": sampling_n,
            "reasoning_effort": reasoning_effort,
            "sampling_temperature": sampling_temperature,
            "sampling_top_p": sampling_top_p,
            "sampling_max_tokens": sampling_max_tokens,
            "post_edit_temperature": post_edit_temperature,
            "post_edit_top_p": post_edit_top_p,
            "post_edit_max_tokens": post_edit_max_tokens,
            "retry": retry,
            "runs": runs,
        },
        "summary": summaries,
        "items": items,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summaries}, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
