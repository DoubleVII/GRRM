import json
import math
from pathlib import Path

from eval.run_oss_diverse_mt_eval import (
    _load_data,
    _parse_data_ids,
    _reference_list,
)
from inference.run_oss_SQM import func_call as run_oss_sqm
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_direct_mt import run_direct_stage


def _mean(values):
    valid = [float(value) for value in values if value is not None and not math.isnan(value)]
    return sum(valid) / len(valid) if valid else None


def main(
    data_id="seedx_challenge_zhen",
    model_path: str = "/home/zfs01/yangs/LLM/openai/gpt-oss-120b",
    output_path: str = "results/oss_direct_mt_eval.json",
    max_samples: int = 16,
    seed: int = 42,
    reasoning_effort: str = "medium",
    temperature: float = 0.3,
    top_p: float = 0.8,
    max_new_tokens: int = 4096,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Run direct gpt-oss translation and reference-aware OSS evaluation only."""
    data_ids = _parse_data_ids(data_id)
    frame = _load_data(data_ids, max_samples, seed)
    print(f"Loaded {len(frame)} items from {', '.join(data_ids)}")

    model = init_oss_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    direct = run_direct_stage(
        frame["src_text"].tolist(),
        frame["src_lang"].tolist(),
        frame["trg_lang"].tolist(),
        model=model,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        retry=retry,
    )

    scores = [None] * len(frame)
    evaluator_responses = [None] * len(frame)
    valid_indices = [
        index for index, translation in enumerate(direct["translations"])
        if translation
    ]
    if valid_indices:
        references = _reference_list(frame)
        evaluation = run_oss_sqm(
            src_list=[frame.iloc[index]["src_text"] for index in valid_indices],
            mt_list=[direct["translations"][index] for index in valid_indices],
            src_langs=[frame.iloc[index]["src_lang"] for index in valid_indices],
            trg_langs=[frame.iloc[index]["trg_lang"] for index in valid_indices],
            ref_list=[references[index] for index in valid_indices],
            model=model,
            model_path=model_path,
        )
        for local_index, original_index in enumerate(valid_indices):
            scores[original_index] = evaluation["scores"][local_index]
            evaluator_responses[original_index] = evaluation["response"][local_index]

    summaries = {}
    for current_data_id in (*data_ids, "overall"):
        indices = (
            list(range(len(frame)))
            if current_data_id == "overall"
            else frame.index[frame["data_id"] == current_data_id].tolist()
        )
        summaries[current_data_id] = {
            "item_count": len(indices),
            "direct_mean": _mean([scores[index] for index in indices]),
            "generation_failures": sum(
                not direct["translations"][index] for index in indices
            ),
            "evaluation_failures": sum(scores[index] is None for index in indices),
        }

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
            "direct_translation": direct["translations"][index],
            "direct_response": direct["responses"][index],
            "direct_score": scores[index],
            "direct_evaluator_response": evaluator_responses[index],
        })

    payload = {
        "model_path": model_path,
        "data_ids": data_ids,
        "max_samples_per_dataset": max_samples,
        "seed": seed,
        "settings": {
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "retry": retry,
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
