import json
import math
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from inference.run_oss_SQM import func_call as run_oss_sqm
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_diverse_mt import run_pipeline
from utils.config import MT_TEST_DATA_META_INFO


def _parse_data_ids(data_id) -> tuple[str, ...]:
    if isinstance(data_id, str):
        values = tuple(value.strip() for value in data_id.split(",") if value.strip())
    elif isinstance(data_id, Iterable):
        values = tuple(data_id)
    else:
        values = (data_id,)
    if not values:
        raise ValueError("At least one data_id is required")
    return values


def _load_data(data_ids: tuple[str, ...], max_samples: int, seed: int) -> pd.DataFrame:
    frames = []
    for data_id in data_ids:
        if data_id not in MT_TEST_DATA_META_INFO:
            raise ValueError(f"Unknown data_id: {data_id}")
        path = Path(MT_TEST_DATA_META_INFO[data_id]["path"])
        frame = pd.read_parquet(path)
        required = {"src_text", "trg_text", "src_lang", "trg_lang"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{data_id} is missing columns: {missing}")
        if max_samples > 0 and len(frame) > max_samples:
            frame = frame.sample(n=max_samples, random_state=seed)
        frame = frame.reset_index().rename(columns={"index": "source_index"})
        frame["data_id"] = data_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _reference_list(frame: pd.DataFrame) -> list[str]:
    references = frame["trg_text"].tolist()
    if "comment" not in frame.columns:
        return references
    return [
        f"{reference}\n评估重点：\n{comment}"
        if pd.notna(comment) and str(comment).strip()
        else reference
        for reference, comment in zip(references, frame["comment"].tolist())
    ]


def _score_translations(
    frame: pd.DataFrame,
    translations: list[Optional[str]],
    model,
    model_path: str,
) -> tuple[list[Optional[float]], list[Optional[str]]]:
    valid_indices = [
        index for index, translation in enumerate(translations) if translation
    ]
    scores = [None] * len(frame)
    responses = [None] * len(frame)
    if not valid_indices:
        return scores, responses

    references = _reference_list(frame)
    result = run_oss_sqm(
        src_list=[frame.iloc[index]["src_text"] for index in valid_indices],
        mt_list=[translations[index] for index in valid_indices],
        src_langs=[frame.iloc[index]["src_lang"] for index in valid_indices],
        trg_langs=[frame.iloc[index]["trg_lang"] for index in valid_indices],
        ref_list=[references[index] for index in valid_indices],
        model=model,
        model_path=model_path,
    )
    for local_index, original_index in enumerate(valid_indices):
        scores[original_index] = result["scores"][local_index]
        responses[original_index] = result["response"][local_index]
    return scores, responses


def _mean(values: list[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None and not math.isnan(value)]
    return sum(valid) / len(valid) if valid else None


def _diversity_stats(analyses: list, prompt_type: str) -> dict:
    segment_counts = []
    candidate_counts = []
    duplicate_counts = 0
    total_candidates = 0
    for analysis in analyses:
        if not analysis or not isinstance(analysis, dict) or "segments" not in analysis:
            continue
        segments = analysis["segments"]
        segment_counts.append(len(segments))
        for segment in segments:
            candidates = [candidate["translation"].strip() for candidate in segment["candidates"]]
            candidate_counts.append(len(candidates))
            normalized = {candidate.casefold() for candidate in candidates}
            duplicate_counts += len(candidates) - len(normalized)
            total_candidates += len(candidates)
    return {
        "prompt_type": prompt_type,
        "valid_stage1": sum(analysis is not None for analysis in analyses),
        "failed_stage1": sum(analysis is None for analysis in analyses),
        "mean_stage1_chars": _mean([
            len(analysis) if isinstance(analysis, str)
            else len(json.dumps(analysis, ensure_ascii=False))
            for analysis in analyses if analysis is not None
        ]),
        "mean_segments_per_item": _mean(segment_counts) if prompt_type == "json" else None,
        "mean_candidates_per_segment": _mean(candidate_counts) if prompt_type == "json" else None,
        "segments_below_three_candidates": (
            sum(count < 3 for count in candidate_counts) if prompt_type == "json" else None
        ),
        "exact_duplicate_candidate_rate": (
            duplicate_counts / total_candidates
            if prompt_type == "json" and total_candidates
            else None
        ),
    }


def _summary_for_indices(
    indices: list[int],
    scores: list[Optional[float]],
    translations: list[Optional[str]],
) -> dict:
    return {
        "item_count": len(indices),
        "two_stage_mean": _mean([scores[index] for index in indices]),
        "generation_failures": sum(not translations[index] for index in indices),
        "evaluation_failures": sum(scores[index] is None for index in indices),
    }


def main(
    data_id="seedx_challenge_zhen",
    model_path: str = "/home/zfs01/yangs/LLM/openai/gpt-oss-120b",
    output_path: str = "results/oss_diverse_mt_eval.json",
    max_samples: int = 16,
    seed: int = 42,
    reasoning_effort: str = "medium",
    min_candidates: int = 3,
    max_candidates: int = 6,
    prompt_type: str = "json",
    divergent_temperature: float = 0.8,
    divergent_top_p: float = 0.95,
    final_temperature: float = 0.3,
    final_top_p: float = 0.8,
    stage1_max_tokens: int = 8192,
    final_max_tokens: int = 4096,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Generate and evaluate divergent/convergent translations only."""
    if prompt_type not in {"json", "codeblock"}:
        raise ValueError("prompt_type must be one of: json, codeblock")
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
        reasoning_effort=reasoning_effort,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        prompt_type=prompt_type,
        divergent_temperature=divergent_temperature,
        divergent_top_p=divergent_top_p,
        final_temperature=final_temperature,
        final_top_p=final_top_p,
        stage1_max_tokens=stage1_max_tokens,
        final_max_tokens=final_max_tokens,
        retry=retry,
    )
    translations = pipeline["convergent"]["translations"]
    scores, evaluator_responses = _score_translations(
        frame, translations, model, model_path
    )

    summaries = {}
    for current_data_id in data_ids:
        indices = frame.index[frame["data_id"] == current_data_id].tolist()
        summaries[current_data_id] = _summary_for_indices(
            indices, scores, translations
        )
    summaries["overall"] = _summary_for_indices(
        list(range(len(frame))), scores, translations
    )

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
            "divergent_analysis": pipeline["divergent"]["analyses"][index],
            "divergent_response": pipeline["divergent"]["responses"][index],
            "two_stage_translation": translations[index],
            "two_stage_response": pipeline["convergent"]["responses"][index],
            "two_stage_score": scores[index],
            "two_stage_evaluator_response": evaluator_responses[index],
        })

    payload = {
        "model_path": model_path,
        "data_ids": data_ids,
        "max_samples_per_dataset": max_samples,
        "seed": seed,
        "settings": {
            "reasoning_effort": reasoning_effort,
            "min_candidates": min_candidates,
            "max_candidates": max_candidates,
            "prompt_type": prompt_type,
            "divergent_temperature": divergent_temperature,
            "divergent_top_p": divergent_top_p,
            "final_temperature": final_temperature,
            "final_top_p": final_top_p,
            "stage1_max_tokens": stage1_max_tokens,
            "final_max_tokens": final_max_tokens,
            "retry": retry,
        },
        "diversity": _diversity_stats(
            pipeline["divergent"]["analyses"], prompt_type
        ),
        "summary": summaries,
        "items": items,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"diversity": payload["diversity"], "summary": summaries}, indent=2))
    print(f"Saved detailed results to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
