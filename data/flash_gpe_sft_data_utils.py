from typing import Any, Optional

from data.oss_mt_sft_data_utils import nonempty_text
from inference.oss_flash_gpe_prompts import validate_prompt_type
from inference.run_oss_flash_gpe_mt import extract_candidate_response
from inference.run_oss_group_post_edit import extract_response as extract_gpe_response


def validate_flash_gpe_record(
    *,
    candidate_response: Optional[str],
    candidate_thinking: Optional[str],
    candidates: list,
    post_edit_response: Optional[str],
    post_edit_thinking: Optional[str],
    post_edit_translation: Optional[str],
    max_candidates: int,
    prompt_type: str,
) -> bool:
    validate_prompt_type(prompt_type, max_candidates)
    parsed_candidates = (
        extract_candidate_response(
            candidate_response,
            max_candidates,
            exact_count=prompt_type in {"markdown", "fixed_4", "fixed_16"},
        )
        if nonempty_text(candidate_response)
        else None
    )
    parsed_post_edit = (
        extract_gpe_response(post_edit_response)
        if nonempty_text(post_edit_response)
        else None
    )
    return (
        nonempty_text(candidate_thinking)
        and isinstance(candidates, list)
        and parsed_candidates == candidates
        and nonempty_text(post_edit_thinking)
        and nonempty_text(parsed_post_edit)
        and parsed_post_edit == post_edit_translation
    )


def normalize_flash_gpe_row(row: Any) -> dict:
    if "flash_gpe_parser_valid" in row.index:
        return {
            "prompt_type": row["flash_gpe_prompt_type"],
            "max_candidates": int(row.get("flash_gpe_target_candidate_count", row["flash_gpe_max_candidates"])),
            "candidate_prompt": row["flash_gpe_stage1_prompt"],
            "candidate_thinking": row["flash_gpe_stage1_thinking"],
            "candidate_response": row["flash_gpe_stage1_response"],
            "candidates": list(row["flash_gpe_candidates"]),
            "post_edit_prompt": row["flash_gpe_stage2_prompt"],
            "post_edit_thinking": row["flash_gpe_stage2_thinking"],
            "post_edit_response": row["flash_gpe_stage2_response"],
            "translation": row["flash_gpe_translation"],
            "parser_valid": bool(row["flash_gpe_parser_valid"]),
            "legacy": False,
        }
    if row.get("gpe_candidate_generation") != "single_call":
        raise ValueError(
            "Row is neither FlashGPE data nor legacy single-call GPE data"
        )
    thinking = list(row["gpe_stage1_thinking"])
    responses = list(row["gpe_stage1_responses"])
    prompts = list(row.get("gpe_stage1_prompts", []))
    if len(thinking) != 1 or len(responses) != 1:
        raise ValueError(
            "Legacy single-call row must have one Stage 1 thinking and response"
        )
    return {
        "prompt_type": row.get("gpe_prompt_type", "fixed_4"),
        "max_candidates": int(
            row.get("gpe_sampling_n", len(row["gpe_stage1_translations"]))
        ),
        "candidate_prompt": prompts[0] if len(prompts) == 1 else None,
        "candidate_thinking": thinking[0],
        "candidate_response": responses[0],
        "candidates": list(row["gpe_stage1_translations"]),
        "post_edit_prompt": row.get("gpe_stage2_prompt"),
        "post_edit_thinking": row["gpe_stage2_thinking"],
        "post_edit_response": row["gpe_stage2_response"],
        "translation": row["gpe_translation"],
        "parser_valid": bool(row["gpe_parser_valid"]),
        "legacy": True,
    }
