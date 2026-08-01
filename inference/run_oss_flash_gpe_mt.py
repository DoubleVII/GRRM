import json
from pathlib import Path
from typing import Optional, Union

from inference.oss_flash_gpe_prompts import (
    build_candidate_prompt,
    build_post_edit_prompt,
    validate_prompt_type,
)
from inference.run_oss_SQM import init_oss_model, load_encoding
from inference.run_oss_diverse_mt import (
    _generate_with_retries,
    _prepare_inputs,
    extract_json_object,
)
from inference.run_oss_group_post_edit import func_call as run_group_post_edit


def extract_candidate_response(
    response: str,
    max_candidates: int,
    *,
    exact_count: bool = False,
) -> Optional[list[str]]:
    value = extract_json_object(response)
    translations = value.get("translations") if isinstance(value, dict) else None
    if not isinstance(translations, list):
        return None
    if not 2 <= len(translations) <= max_candidates:
        return None
    if exact_count and len(translations) != max_candidates:
        return None
    if any(not isinstance(item, str) or not item.strip() for item in translations):
        return None
    translations = [item.strip() for item in translations]
    normalized = {" ".join(item.split()).casefold() for item in translations}
    if len(normalized) != len(translations):
        return None
    return translations


def _normalize_languages(
    item_count: int,
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
) -> tuple[list[str], list[str]]:
    src_langs = (
        [src_langs] * item_count if isinstance(src_langs, str) else src_langs
    )
    trg_langs = (
        [trg_langs] * item_count if isinstance(trg_langs, str) else trg_langs
    )
    if not (item_count == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")
    return src_langs, trg_langs


def run_candidate_generation_stage(
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    max_candidates: int = 4,
    prompt_type: str = "fixed_4",
    model=None,
    model_path: str = "openai/gpt-oss-120b",
    reasoning_effort: Optional[str] = "medium",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    validate_prompt_type(prompt_type, max_candidates)
    src_langs, trg_langs = _normalize_languages(
        len(src_list), src_langs, trg_langs
    )
    llm = init_oss_model(model_path) if model is None else model
    encoding = load_encoding()
    prompts = [
        build_candidate_prompt(
            src_lang,
            trg_lang,
            source,
            max_candidates,
            prompt_type=prompt_type,
        )
        for source, src_lang, trg_lang in zip(src_list, src_langs, trg_langs)
    ]
    results = _generate_with_retries(
        llm,
        _prepare_inputs(prompts, encoding, reasoning_effort),
        lambda response: extract_candidate_response(
            response,
            max_candidates,
            exact_count=prompt_type == "fixed_4",
        ),
        encoding,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        retry=retry,
    )
    return {
        "prompts": prompts,
        "translations": [result["parsed"] or [] for result in results],
        "responses": [result["response"] for result in results],
        "thinking": [result["thinking"] for result in results],
    }


def run_pipeline(
    src_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    *,
    model,
    model_path: str,
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
) -> dict:
    validate_prompt_type(prompt_type, max_candidates)
    candidate_generation = run_candidate_generation_stage(
        src_list,
        src_langs,
        trg_langs,
        max_candidates=max_candidates,
        prompt_type=prompt_type,
        model=model,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        temperature=candidate_temperature,
        top_p=candidate_top_p,
        max_tokens=candidate_max_tokens,
        retry=retry,
    )
    candidates = candidate_generation["translations"]
    valid_indices = [
        index for index, values in enumerate(candidates) if len(values) >= 2
    ]
    post_edit = {
        "translations": [None] * len(src_list),
        "responses": [None] * len(src_list),
        "thinking": [None] * len(src_list),
    }
    if valid_indices:
        result = run_group_post_edit(
            src_list=[src_list[index] for index in valid_indices],
            mt_list=[candidates[index] for index in valid_indices],
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
            prompt_builder=build_post_edit_prompt,
        )
        for local_index, original_index in enumerate(valid_indices):
            post_edit["translations"][original_index] = result["post_edit_mt"][
                local_index
            ]
            post_edit["responses"][original_index] = result["response"][
                local_index
            ]
            post_edit["thinking"][original_index] = result["thinking"][
                local_index
            ]
    return {
        "candidate_generation": candidate_generation,
        "usable_candidate_counts": [len(values) for values in candidates],
        "post_edit": post_edit,
    }


def main(
    input_path: str,
    output_path: str,
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
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
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Run one-call candidate generation followed by OSS post-editing."""
    import pandas as pd

    validate_prompt_type(prompt_type, max_candidates)
    frame = pd.read_parquet(input_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
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
    items = []
    for index, row in frame.iterrows():
        items.append({
            "index": index,
            "src_text": row["src_text"],
            "ref_text": row.get("trg_text"),
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "candidates": output["candidate_generation"]["translations"][index],
            "candidate_prompt": output["candidate_generation"]["prompts"][index],
            "candidate_response": output["candidate_generation"]["responses"][index],
            "candidate_thinking": output["candidate_generation"]["thinking"][index],
            "usable_candidate_count": output["usable_candidate_counts"][index],
            "flash_gpe_translation": output["post_edit"]["translations"][index],
            "flash_gpe_response": output["post_edit"]["responses"][index],
            "flash_gpe_thinking": output["post_edit"]["thinking"][index],
        })
    payload = {
        "method": "flash_gpe",
        "model_path": model_path,
        "input_path": input_path,
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
