import json
from pathlib import Path
from typing import Optional, Union

from utils.config import candidate_identifiers

from inference.run_oss_SQM import init_oss_model
from inference.run_oss_direct_mt import run_direct_stage
from inference.run_oss_group_post_edit import func_call as run_group_post_edit


def validate_sampling_n(sampling_n: int) -> None:
    if not 2 <= sampling_n <= len(candidate_identifiers):
        raise ValueError(
            f"sampling_n must be between 2 and {len(candidate_identifiers)}"
        )


def run_direct_sampling_stage(
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    sampling_n: int = 4,
    model=None,
    model_path: str = "openai/gpt-oss-120b",
    reasoning_effort: Optional[str] = "medium",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    validate_sampling_n(sampling_n)
    item_count = len(src_list)
    src_langs = [src_langs] * item_count if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * item_count if isinstance(trg_langs, str) else trg_langs
    if not (item_count == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")

    sampled = run_direct_stage(
        [source for source in src_list for _ in range(sampling_n)],
        [language for language in src_langs for _ in range(sampling_n)],
        [language for language in trg_langs for _ in range(sampling_n)],
        model=model,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        retry=retry,
    )

    def regroup(values: list) -> list[list]:
        return [
            values[index * sampling_n : (index + 1) * sampling_n]
            for index in range(item_count)
        ]

    return {
        "translations": regroup(sampled["translations"]),
        "responses": regroup(sampled["responses"]),
        "thinking": regroup(sampled["thinking"]),
    }


def run_pipeline(
    src_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    *,
    model,
    model_path: str,
    sampling_n: int = 4,
    reasoning_effort: str = "medium",
    sampling_temperature: float = 0.8,
    sampling_top_p: float = 0.95,
    sampling_max_tokens: int = 4096,
    post_edit_temperature: float = 0.3,
    post_edit_top_p: float = 0.8,
    post_edit_max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    sampled = run_direct_sampling_stage(
        src_list,
        src_langs,
        trg_langs,
        sampling_n=sampling_n,
        model=model,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        temperature=sampling_temperature,
        top_p=sampling_top_p,
        max_tokens=sampling_max_tokens,
        retry=retry,
    )

    usable_candidates = [
        [
            candidate
            for candidate in candidates
            if isinstance(candidate, str) and candidate.strip()
        ]
        for candidates in sampled["translations"]
    ]
    valid_indices = [
        index
        for index, candidates in enumerate(usable_candidates)
        if len(candidates) >= 2
    ]
    post_edit = {
        "translations": [None] * len(src_list),
        "responses": [None] * len(src_list),
        "thinking": [None] * len(src_list),
    }
    if valid_indices:
        result = run_group_post_edit(
            src_list=[src_list[index] for index in valid_indices],
            mt_list=[usable_candidates[index] for index in valid_indices],
            notes_list=[None] * len(valid_indices),
            src_langs=[src_langs[index] for index in valid_indices],
            trg_langs=[trg_langs[index] for index in valid_indices],
            temperature=post_edit_temperature,
            top_p=post_edit_top_p,
            retry=retry,
            model=model,
            model_path=model_path,
            reasoning_effort=reasoning_effort,
            max_new_tokens=post_edit_max_tokens,
        )
        for local_index, original_index in enumerate(valid_indices):
            post_edit["translations"][original_index] = result["post_edit_mt"][
                local_index
            ]
            post_edit["responses"][original_index] = result["response"][local_index]
            post_edit["thinking"][original_index] = result["thinking"][local_index]

    return {
        "sampling": sampled,
        "usable_candidate_counts": [len(values) for values in usable_candidates],
        "post_edit": post_edit,
    }


def main(
    input_path: str,
    output_path: str,
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
    sampling_n: int = 4,
    reasoning_effort: str = "medium",
    sampling_temperature: float = 0.8,
    sampling_top_p: float = 0.95,
    sampling_max_tokens: int = 4096,
    post_edit_temperature: float = 0.3,
    post_edit_top_p: float = 0.8,
    post_edit_max_tokens: int = 4096,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Run direct sampling followed by OSS group post-editing."""
    import pandas as pd

    validate_sampling_n(sampling_n)
    frame = pd.read_parquet(input_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)

    model = init_oss_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    output = run_pipeline(
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
    items = []
    for index, row in frame.reset_index(drop=True).iterrows():
        items.append({
            "index": index,
            "src_text": row["src_text"],
            "ref_text": row.get("trg_text"),
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "mt_candidates": output["sampling"]["translations"][index],
            "candidate_responses": output["sampling"]["responses"][index],
            "candidate_thinking": output["sampling"]["thinking"][index],
            "usable_candidate_count": output["usable_candidate_counts"][index],
            "group_post_edit_translation": output["post_edit"]["translations"][index],
            "group_post_edit_response": output["post_edit"]["responses"][index],
            "group_post_edit_thinking": output["post_edit"]["thinking"][index],
        })
    payload = {
        "model_path": model_path,
        "input_path": input_path,
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
        },
        "items": items,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
