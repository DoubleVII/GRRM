import json
import re
from pathlib import Path
from typing import Optional

from eval.run_mt_eval import _release_vllm_model
from eval.run_oss_diverse_mt_eval import (
    _diversity_stats,
    _load_data,
    _parse_data_ids,
    _score_translations,
    _summary_for_indices,
)
from inference.run_oss_SQM import init_oss_model
from inference.run_qwen_sft_mt import (
    first_parsed,
    init_sft_mt_model,
    run_direct_stage,
    run_flash_gpe_pipeline,
    run_gpe_pipeline,
    run_scd_pipeline,
)


SCORE_NAMES = {
    "direct": "direct_mean",
    "flash_gpe": "flash_gpe_mean",
    "group_post_edit": "group_post_edit_mean",
    "scd": "scd_mean",
}


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return label or "model"


def _record(output: dict) -> Optional[dict]:
    return output[0] if output else None


def main(
    method: str,
    model_path: str,
    model_label: Optional[str] = None,
    output_path: Optional[str] = None,
    data_id: str = "seedx_challenge_zhen",
    evaluator_model_path: str = "/home/zfs01/yangs/LLM/openai/gpt-oss-120b",
    max_samples: int = 0,
    seed: int = 42,
    sampling_n: int = 4,
    prompt_type: str = "fixed_4",
    min_candidates: int = 3,
    max_candidates: Optional[int] = None,
    generation_temperature: float = 0.3,
    generation_top_p: float = 0.8,
    sampling_temperature: float = 0.8,
    sampling_top_p: float = 0.95,
    final_temperature: float = 0.3,
    final_top_p: float = 0.8,
    max_tokens: int = 4096,
    stage1_max_tokens: int = 8192,
    retry: int = 3,
    runs: int = 4,
    gpu_memory_utilization: float = 0.9,
    evaluator_gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    method = method.strip().lower()
    if method not in SCORE_NAMES:
        raise ValueError(
            "method must be one of: direct, group_post_edit, flash_gpe, scd"
        )
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if max_candidates is None:
        max_candidates = 4 if method == "flash_gpe" else 6
    label = _safe_label(model_label or Path(model_path).name)
    if output_path is None:
        mode = (
            f".{prompt_type}.max{max_candidates}"
            if method == "flash_gpe" else ""
        )
        output_path = f"results/qwen_sft_mt_eval.{label}.{method}{mode}.json"

    data_ids = _parse_data_ids(data_id)
    frame = _load_data(data_ids, max_samples, seed)
    sources = frame["src_text"].tolist()
    src_langs = frame["src_lang"].tolist()
    trg_langs = frame["trg_lang"].tolist()
    print(f"Loaded {len(frame)} items from {', '.join(data_ids)}")

    engine = init_sft_mt_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_sleep_mode=True,
    )
    if method == "direct":
        pipeline = run_direct_stage(
            sources,
            src_langs,
            trg_langs,
            engine=engine,
            temperature=generation_temperature,
            top_p=generation_top_p,
            max_tokens=max_tokens,
            retry=retry,
        )
        translations = first_parsed(pipeline["outputs"])
    elif method == "group_post_edit":
        pipeline = run_gpe_pipeline(
            sources,
            src_langs,
            trg_langs,
            engine=engine,
            sampling_n=sampling_n,
            sampling_temperature=sampling_temperature,
            sampling_top_p=sampling_top_p,
            sampling_max_tokens=max_tokens,
            post_edit_temperature=final_temperature,
            post_edit_top_p=final_top_p,
            post_edit_max_tokens=max_tokens,
            retry=retry,
        )
        translations = first_parsed(pipeline["post_edit"])
    elif method == "flash_gpe":
        pipeline = run_flash_gpe_pipeline(
            sources,
            src_langs,
            trg_langs,
            engine=engine,
            max_candidates=max_candidates,
            prompt_type=prompt_type,
            candidate_temperature=sampling_temperature,
            candidate_top_p=sampling_top_p,
            candidate_max_tokens=max_tokens,
            post_edit_temperature=final_temperature,
            post_edit_top_p=final_top_p,
            post_edit_max_tokens=max_tokens,
            retry=retry,
        )
        translations = first_parsed(pipeline["post_edit"])
    else:
        pipeline = run_scd_pipeline(
            sources,
            src_langs,
            trg_langs,
            engine=engine,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            stage1_temperature=sampling_temperature,
            stage1_top_p=sampling_top_p,
            stage1_max_tokens=stage1_max_tokens,
            stage2_temperature=final_temperature,
            stage2_top_p=final_top_p,
            stage2_max_tokens=max_tokens,
            retry=retry,
        )
        translations = first_parsed(pipeline["stage2"])

    _release_vllm_model(engine.model)
    del engine
    evaluator = init_oss_model(
        evaluator_model_path,
        gpu_memory_utilization=evaluator_gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    evaluation = _score_translations(
        frame, translations, evaluator, evaluator_model_path, runs=runs
    )
    scores = evaluation["scores"]
    scores_by_run = evaluation["scores_by_run"]

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
            score_name=SCORE_NAMES[method],
        )
        if method in {"group_post_edit", "flash_gpe"}:
            if method == "group_post_edit":
                summary["candidate_generation_failures"] = sum(
                    max(
                        0,
                        sampling_n
                        - pipeline["usable_candidate_counts"][index],
                    )
                    for index in indices
                )
            else:
                summary["candidate_generation_failures"] = sum(
                    pipeline["usable_candidate_counts"][index] < 2
                    for index in indices
                )
            candidate_counts = [
                pipeline["usable_candidate_counts"][index]
                for index in indices
            ]
            summary["candidate_count_mean"] = (
                sum(candidate_counts) / len(candidate_counts)
                if candidate_counts else None
            )
            summary["candidate_count_min"] = min(candidate_counts, default=None)
            summary["candidate_count_max"] = max(candidate_counts, default=None)
            summary["candidate_count_distribution"] = {
                str(count): candidate_counts.count(count)
                for count in sorted(set(candidate_counts))
            }
        if method == "scd":
            summary["stage1_failures"] = sum(
                not pipeline["stage1"][index] for index in indices
            )
            summary["stage2_failures"] = sum(
                not pipeline["stage2"][index] for index in indices
            )
        summaries[current_data_id] = summary

    items = []
    for index, row in frame.iterrows():
        item = {
            "index": index,
            "data_id": row["data_id"],
            "source_index": int(row["source_index"]),
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "src_text": row["src_text"],
            "ref_text": row["trg_text"],
            "translation": translations[index],
            "score": scores[index],
            "scores": [values[index] for values in scores_by_run],
            "evaluator_responses": [
                values[index] for values in evaluation["responses_by_run"]
            ],
        }
        if method == "direct":
            item["direct"] = _record(pipeline["outputs"][index])
        elif method == "group_post_edit":
            item.update({
                "direct_candidates": pipeline["sampling"]["outputs"][index],
                "candidates": pipeline["candidates"][index],
                "usable_candidate_count": pipeline["usable_candidate_counts"][index],
                "post_edit": _record(pipeline["post_edit"][index]),
            })
        elif method == "flash_gpe":
            item.update({
                "candidate_generation": _record(
                    pipeline["candidate_generation"]["outputs"][index]
                ),
                "candidates": pipeline["candidates"][index],
                "usable_candidate_count": pipeline["usable_candidate_counts"][index],
                "flash_gpe": _record(pipeline["post_edit"][index]),
            })
        else:
            item.update({
                "stage1": _record(pipeline["stage1"][index]),
                "stage2": _record(pipeline["stage2"][index]),
            })
        items.append(item)

    payload = {
        "method": method,
        "model_path": model_path,
        "model_label": label,
        "evaluator_model_path": evaluator_model_path,
        "data_ids": data_ids,
        "settings": {
            "sampling_n": sampling_n,
            "min_candidates": min_candidates,
            "max_candidates": max_candidates,
            "generation_temperature": generation_temperature,
            "generation_top_p": generation_top_p,
            "sampling_temperature": sampling_temperature,
            "sampling_top_p": sampling_top_p,
            "final_temperature": final_temperature,
            "final_top_p": final_top_p,
            "max_tokens": max_tokens,
            "stage1_max_tokens": stage1_max_tokens,
            "retry": retry,
            "runs": runs,
        },
        "summary": summaries,
        "items": items,
    }
    if method == "flash_gpe":
        payload["settings"].update({
            "max_candidates": max_candidates,
            "prompt_type": prompt_type,
        })
    if method == "scd":
        payload["diversity"] = _diversity_stats(
            [values[0]["parsed"] if values else None for values in pipeline["stage1"]],
            "json",
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"diversity": payload.get("diversity"), "summary": summaries}, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
