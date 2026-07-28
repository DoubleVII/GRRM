import json
from typing import Any, Optional

from inference.run_oss_diverse_mt import (
    extract_codeblock_response,
    extract_final_translation,
    extract_json_object,
    validate_divergent_result,
)
from inference.run_oss_group_post_edit import extract_response as extract_gpe_response


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_scd_stage1_response(
    response: Optional[str],
    *,
    prompt_type: str,
    candidate_confidence: bool,
):
    if not nonempty_text(response):
        return None
    if prompt_type == "json":
        return validate_divergent_result(
            extract_json_object(response),
            candidate_confidence=candidate_confidence,
        )
    if prompt_type == "codeblock":
        return extract_codeblock_response(response)
    raise ValueError("SCD SFT collection supports prompt_type=json or codeblock")


def validate_scd_record(
    *,
    stage1_response: Optional[str],
    stage1_thinking: Optional[str],
    stage1_analysis,
    stage2_response: Optional[str],
    stage2_thinking: Optional[str],
    stage2_translation: Optional[str],
    prompt_type: str,
    candidate_confidence: bool,
) -> bool:
    parsed_stage1 = parse_scd_stage1_response(
        stage1_response,
        prompt_type=prompt_type,
        candidate_confidence=candidate_confidence,
    )
    parsed_stage2 = (
        extract_final_translation(stage2_response)
        if nonempty_text(stage2_response)
        else None
    )
    return (
        nonempty_text(stage1_thinking)
        and nonempty_text(stage2_thinking)
        and parsed_stage1 is not None
        and parsed_stage1 == stage1_analysis
        and nonempty_text(parsed_stage2)
        and parsed_stage2 == stage2_translation
    )


def validate_gpe_record(
    *,
    candidate_responses: list,
    candidate_thinking: list,
    candidate_translations: list,
    post_edit_response: Optional[str],
    post_edit_thinking: Optional[str],
    post_edit_translation: Optional[str],
    sampling_n: int,
) -> bool:
    if not (
        isinstance(candidate_responses, list)
        and isinstance(candidate_thinking, list)
        and isinstance(candidate_translations, list)
        and len(candidate_responses) == sampling_n
        and len(candidate_thinking) == sampling_n
        and len(candidate_translations) == sampling_n
    ):
        return False
    for response, thinking, translation in zip(
        candidate_responses, candidate_thinking, candidate_translations
    ):
        parsed = (
            extract_final_translation(response) if nonempty_text(response) else None
        )
        if not (
            nonempty_text(thinking)
            and nonempty_text(parsed)
            and parsed == translation
        ):
            return False
    parsed_post_edit = (
        extract_gpe_response(post_edit_response)
        if nonempty_text(post_edit_response)
        else None
    )
    return (
        nonempty_text(post_edit_thinking)
        and nonempty_text(parsed_post_edit)
        and parsed_post_edit == post_edit_translation
    )


def serialize_parsed(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
