import json
import re
from pathlib import Path
from typing import Optional

from inference.inst_flash_gqm_prompts import (
    build_gqm_prompt,
    validate_candidate_count,
)
from inference.run_inst_flash_gpe_mt import (
    _generate_with_retries,
    _normalize_languages,
    init_inst_model,
    run_candidate_generation_stage,
)
from utils.config import candidate_identifiers


def _parse_ranking(ranking_text: str, expected_identifiers: list[str]):
    if "<" in ranking_text or "\n" in ranking_text:
        return None
    raw_tiers = ranking_text.split(">")
    if not raw_tiers or any(not tier.strip() for tier in raw_tiers):
        return None
    tiers = []
    for raw_tier in raw_tiers:
        values = [value.strip() for value in raw_tier.split("=")]
        if any(not re.fullmatch(r"[A-H]", value) for value in values):
            return None
        tiers.append(values)
    flattened = [value for tier in tiers for value in tier]
    if len(flattened) != len(set(flattened)):
        return None
    if set(flattened) != set(expected_identifiers):
        return None
    return tiers


def _parse_scores(score_text: str, expected_identifiers: list[str]):
    if "\n" in score_text:
        return None
    score_map = {}
    for item in score_text.split(","):
        match = re.fullmatch(r"\s*([A-H])\s*:\s*(10|[0-9])\s*", item)
        if match is None or match.group(1) in score_map:
            return None
        score_map[match.group(1)] = int(match.group(2))
    if set(score_map) != set(expected_identifiers):
        return None
    return score_map


def extract_gqm_response(
    response: str,
    candidate_count: int,
    *,
    explicit_analysis: bool = False,
) -> Optional[dict]:
    if not isinstance(response, str):
        return None
    try:
        validate_candidate_count(candidate_count)
    except ValueError:
        return None

    response = response.strip()
    analysis_marker = "# Step-by-step Analysis"
    ranking_marker = "# Final Ranking"
    scores_marker = "# Scores"
    if response.count(ranking_marker) != 1 or response.count(scores_marker) != 1:
        return None
    ranking_index = response.index(ranking_marker)
    scores_index = response.index(scores_marker)
    if ranking_index >= scores_index:
        return None

    prefix = response[:ranking_index].strip()
    if explicit_analysis:
        if not prefix.startswith(analysis_marker) or prefix.count(analysis_marker) != 1:
            return None
        if not prefix[len(analysis_marker):].strip():
            return None
    elif prefix:
        return None

    ranking_text = response[
        ranking_index + len(ranking_marker):scores_index
    ].strip()
    score_text = response[scores_index + len(scores_marker):].strip()
    expected = candidate_identifiers[:candidate_count]
    tiers = _parse_ranking(ranking_text, expected)
    score_map = _parse_scores(score_text, expected)
    if tiers is None or score_map is None:
        return None

    previous_score = None
    for tier in tiers:
        tier_scores = {score_map[identifier] for identifier in tier}
        if len(tier_scores) != 1:
            return None
        current_score = next(iter(tier_scores))
        if previous_score is not None and current_score >= previous_score:
            return None
        previous_score = current_score

    return {
        "ranking": ranking_text,
        "scores": [score_map[identifier] for identifier in expected],
    }


def run_gqm_ranking_stage(
    src_list,
    src_langs,
    trg_langs,
    candidate_lists,
    *,
    model,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    max_tokens=8192,
    retry=3,
    enable_thinking=True,
):
    size = len(src_list)
    src_langs, trg_langs = _normalize_languages(size, src_langs, trg_langs)
    if len(candidate_lists) != size:
        raise ValueError("candidate_lists must match src_list length")
    for candidates in candidate_lists:
        validate_candidate_count(len(candidates))

    prompts = [
        build_gqm_prompt(
            source_lang,
            target_lang,
            source,
            candidates,
            explicit_analysis=not enable_thinking,
        )
        for source, source_lang, target_lang, candidates in zip(
            src_list, src_langs, trg_langs, candidate_lists
        )
    ]
    results = _generate_with_retries(
        model,
        prompts,
        lambda text, index: extract_gqm_response(
            text,
            len(candidate_lists[index]),
            explicit_analysis=not enable_thinking,
        ),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        retry=retry,
        enable_thinking=enable_thinking,
    )

    rankings = []
    scores = []
    selected_indices = []
    translations = []
    for candidates, result in zip(candidate_lists, results):
        parsed = result["parsed"]
        if parsed is None:
            rankings.append(None)
            scores.append(None)
            selected_indices.append(None)
            translations.append(None)
            continue
        item_scores = parsed["scores"]
        selected_index = max(range(len(item_scores)), key=item_scores.__getitem__)
        rankings.append(parsed["ranking"])
        scores.append(item_scores)
        selected_indices.append(selected_index)
        translations.append(candidates[selected_index])

    return {
        "prompts": prompts,
        "rankings": rankings,
        "scores": scores,
        "responses": [result["response"] for result in results],
        "raw_outputs": [result["raw_output"] for result in results],
        "thinking": [result["thinking"] for result in results],
        "output_tokens": [result["output_tokens"] for result in results],
        "selected_candidate_indices": selected_indices,
        "translations": translations,
    }


def run_pipeline(
    src_list,
    src_langs,
    trg_langs,
    *,
    model,
    max_candidates=8,
    candidate_counts=None,
    candidate_temperature=1.0,
    candidate_top_p=1.0,
    candidate_top_k=0,
    candidate_presence_penalty=0.0,
    candidate_repetition_penalty=1.0,
    candidate_max_tokens=8192,
    gqm_temperature=1.0,
    gqm_top_p=1.0,
    gqm_top_k=0,
    gqm_presence_penalty=0.0,
    gqm_repetition_penalty=1.0,
    gqm_max_tokens=8192,
    retry=3,
    enable_thinking=True,
):
    validate_candidate_count(max_candidates)
    src_langs, trg_langs = _normalize_languages(len(src_list), src_langs, trg_langs)
    candidate = run_candidate_generation_stage(
        src_list,
        src_langs,
        trg_langs,
        max_candidates=max_candidates,
        candidate_counts=candidate_counts,
        model=model,
        temperature=candidate_temperature,
        top_p=candidate_top_p,
        top_k=candidate_top_k,
        presence_penalty=candidate_presence_penalty,
        repetition_penalty=candidate_repetition_penalty,
        max_tokens=candidate_max_tokens,
        retry=retry,
        enable_thinking=enable_thinking,
    )
    valid_indices = [
        index
        for index, candidates in enumerate(candidate["translations"])
        if len(candidates) >= 2
    ]
    gqm = {
        "prompts": [None] * len(src_list),
        "rankings": [None] * len(src_list),
        "scores": [None] * len(src_list),
        "responses": [None] * len(src_list),
        "raw_outputs": [None] * len(src_list),
        "thinking": [None] * len(src_list),
        "output_tokens": [None] * len(src_list),
        "selected_candidate_indices": [None] * len(src_list),
        "translations": [None] * len(src_list),
    }
    if valid_indices:
        ranked = run_gqm_ranking_stage(
            [src_list[index] for index in valid_indices],
            [src_langs[index] for index in valid_indices],
            [trg_langs[index] for index in valid_indices],
            [candidate["translations"][index] for index in valid_indices],
            model=model,
            temperature=gqm_temperature,
            top_p=gqm_top_p,
            top_k=gqm_top_k,
            presence_penalty=gqm_presence_penalty,
            repetition_penalty=gqm_repetition_penalty,
            max_tokens=gqm_max_tokens,
            retry=retry,
            enable_thinking=enable_thinking,
        )
        for local_index, original_index in enumerate(valid_indices):
            for key in gqm:
                gqm[key][original_index] = ranked[key][local_index]

    return {
        "candidate_generation": candidate,
        "usable_candidate_counts": [
            len(candidates) for candidates in candidate["translations"]
        ],
        "gqm": gqm,
    }


def main(
    input_path: str,
    output_path: str,
    model_path: str,
    max_samples: int = 0,
    max_candidates: int = 8,
    candidate_temperature: float = 1.0,
    candidate_top_p: float = 1.0,
    candidate_top_k: int = 0,
    candidate_presence_penalty: float = 0.0,
    candidate_repetition_penalty: float = 1.0,
    candidate_max_tokens: int = 8192,
    gqm_temperature: float = 1.0,
    gqm_top_p: float = 1.0,
    gqm_top_k: int = 0,
    gqm_presence_penalty: float = 0.0,
    gqm_repetition_penalty: float = 1.0,
    gqm_max_tokens: int = 8192,
    retry: int = 3,
    enable_thinking: bool = True,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    import pandas as pd

    frame = pd.read_parquet(input_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    engine = init_inst_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    result = run_pipeline(
        frame.src_text.tolist(),
        frame.src_lang.tolist(),
        frame.trg_lang.tolist(),
        model=engine,
        max_candidates=max_candidates,
        candidate_temperature=candidate_temperature,
        candidate_top_p=candidate_top_p,
        candidate_top_k=candidate_top_k,
        candidate_presence_penalty=candidate_presence_penalty,
        candidate_repetition_penalty=candidate_repetition_penalty,
        candidate_max_tokens=candidate_max_tokens,
        gqm_temperature=gqm_temperature,
        gqm_top_p=gqm_top_p,
        gqm_top_k=gqm_top_k,
        gqm_presence_penalty=gqm_presence_penalty,
        gqm_repetition_penalty=gqm_repetition_penalty,
        gqm_max_tokens=gqm_max_tokens,
        retry=retry,
        enable_thinking=enable_thinking,
    )
    items = []
    for index, row in frame.iterrows():
        items.append({
            "index": index,
            "src_text": row.src_text,
            "ref_text": row.get("trg_text"),
            "src_lang": row.src_lang,
            "trg_lang": row.trg_lang,
            "candidates": result["candidate_generation"]["translations"][index],
            "candidate_prompt": result["candidate_generation"]["prompts"][index],
            "candidate_response": result["candidate_generation"]["responses"][index],
            "candidate_thinking": result["candidate_generation"]["thinking"][index],
            "candidate_output_tokens": result["candidate_generation"]["output_tokens"][index],
            "usable_candidate_count": result["usable_candidate_counts"][index],
            "gqm_prompt": result["gqm"]["prompts"][index],
            "gqm_ranking": result["gqm"]["rankings"][index],
            "gqm_scores": result["gqm"]["scores"][index],
            "gqm_response": result["gqm"]["responses"][index],
            "gqm_thinking": result["gqm"]["thinking"][index],
            "gqm_output_tokens": result["gqm"]["output_tokens"][index],
            "selected_candidate_index": result["gqm"]["selected_candidate_indices"][index],
            "flash_gqm_translation": result["gqm"]["translations"][index],
        })
    payload = {
        "method": "inst_flash_gqm",
        "model_path": model_path,
        "settings": {
            "prompt_type": "markdown",
            "max_candidates": max_candidates,
            "enable_thinking": enable_thinking,
            "retry": retry,
            "tie_break": "first_maximum",
            "candidate_temperature": candidate_temperature,
            "candidate_top_p": candidate_top_p,
            "candidate_top_k": candidate_top_k,
            "candidate_presence_penalty": candidate_presence_penalty,
            "candidate_repetition_penalty": candidate_repetition_penalty,
            "candidate_max_tokens": candidate_max_tokens,
            "gqm_temperature": gqm_temperature,
            "gqm_top_p": gqm_top_p,
            "gqm_top_k": gqm_top_k,
            "gqm_presence_penalty": gqm_presence_penalty,
            "gqm_repetition_penalty": gqm_repetition_penalty,
            "gqm_max_tokens": gqm_max_tokens,
        },
        "items": items,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(items)} items to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
