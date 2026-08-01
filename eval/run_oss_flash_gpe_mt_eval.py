import json
from pathlib import Path
from typing import Optional

from eval.run_oss_diverse_mt_eval import (
    _load_data,
    _parse_data_ids,
    _score_translations,
    _summary_for_indices,
)
from inference.oss_flash_gpe_prompts import validate_prompt_type
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_flash_gpe_mt import run_pipeline


def main(
    data_id="seedx_challenge_zhen",
    model_path: str = "/home/zfs01/yangs/LLM/openai/gpt-oss-120b",
    output_path: Optional[str] = None,
    max_samples: int = 16,
    seed: int = 42,
    max_candidates: int = 4,
    prompt_type: str = "fixed_4",
    reasoning_effort: str = "medium",
    candidate_temperature: float = 0.8,
    candidate_top_p: float = 0.95,
    candidate_max_tokens: int = 4096,
    post_edit_temperature: float = 0.3,
    post_edit_top_p: float = 0.8,
    post_edit_max_tokens: int = 4096,
    retry: int = 3,
    runs: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Generate and evaluate FlashGPE translations with gpt-oss."""
    validate_prompt_type(prompt_type, max_candidates)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if output_path is None:
        output_path = (
            "results/oss_flash_gpe_mt_eval."
            f"{prompt_type}.max{max_candidates}.json"
        )
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
        max_candidates=max_candidates,
        prompt_type=prompt_type,
        reasoning_effort=reasoning_effort,
        candidate_temperature=candidate_temperature,
        candidate_top_p=candidate_top_p,
        candidate_max_tokens=candidate_max_tokens,
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
    responses_by_run = evaluation["responses_by_run"]
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
            score_name="flash_gpe_mean",
        )
        summary["candidate_generation_failures"] = sum(
            pipeline["usable_candidate_counts"][index] < 2
            for index in indices
        )
        counts = [
            pipeline["usable_candidate_counts"][index] for index in indices
        ]
        summary["candidate_count_mean"] = (
            sum(counts) / len(counts) if counts else None
        )
        summary["candidate_count_min"] = min(counts, default=None)
        summary["candidate_count_max"] = max(counts, default=None)
        summary["candidate_count_distribution"] = {
            str(count): counts.count(count) for count in sorted(set(counts))
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
            "candidates": pipeline["candidate_generation"]["translations"][index],
            "candidate_response": pipeline["candidate_generation"]["responses"][index],
            "candidate_thinking": pipeline["candidate_generation"]["thinking"][index],
            "usable_candidate_count": pipeline["usable_candidate_counts"][index],
            "flash_gpe_translation": translations[index],
            "flash_gpe_response": pipeline["post_edit"]["responses"][index],
            "flash_gpe_thinking": pipeline["post_edit"]["thinking"][index],
            "flash_gpe_score": scores[index],
            "flash_gpe_scores": [values[index] for values in scores_by_run],
            "flash_gpe_evaluator_responses": [
                values[index] for values in responses_by_run
            ],
        })
    payload = {
        "method": "flash_gpe",
        "model_path": model_path,
        "data_ids": data_ids,
        "max_samples_per_dataset": max_samples,
        "seed": seed,
        "settings": {
            "max_candidates": max_candidates,
            "prompt_type": prompt_type,
            "candidate_count_mode": (
                "fixed" if prompt_type == "fixed_4" else "adaptive_max"
            ),
            "reasoning_effort": reasoning_effort,
            "candidate_temperature": candidate_temperature,
            "candidate_top_p": candidate_top_p,
            "candidate_max_tokens": candidate_max_tokens,
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
