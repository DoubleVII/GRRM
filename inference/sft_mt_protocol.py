import re
from dataclasses import dataclass
from typing import Callable, Optional

from utils.config import LANG_MAP, candidate_identifiers


THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"
RESPONSE_OPEN = "<response>"
RESPONSE_CLOSE = "</response>"

@dataclass(frozen=True)
class SftOutput:
    thinking: str
    response: str


_OUTPUT_PATTERN = re.compile(
    rf"\s*{re.escape(THINKING_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(THINKING_CLOSE)}\s*"
    rf"{re.escape(RESPONSE_OPEN)}\s*(.*?)\s*"
    rf"{re.escape(RESPONSE_CLOSE)}\s*",
    re.DOTALL,
)


def format_sft_output(thinking: str, response: str) -> str:
    thinking = _require_content(thinking, "thinking")
    response = _require_content(response, "response")
    _reject_protocol_tags(thinking, "thinking")
    _reject_protocol_tags(response, "response")
    return (
        f"{THINKING_OPEN}\n{thinking}\n{THINKING_CLOSE}\n"
        f"{RESPONSE_OPEN}\n{response}\n{RESPONSE_CLOSE}"
    )


def parse_sft_output(text: Optional[str]) -> Optional[SftOutput]:
    if not isinstance(text, str):
        return None
    if any(text.count(tag) != 1 for tag in (
        THINKING_OPEN, THINKING_CLOSE, RESPONSE_OPEN, RESPONSE_CLOSE
    )):
        return None
    match = _OUTPUT_PATTERN.fullmatch(text)
    if match is None:
        return None
    thinking, response = (value.strip() for value in match.groups())
    if not thinking or not response:
        return None
    return SftOutput(thinking=thinking, response=response)


def parse_task_output(text: Optional[str], parser: Callable[[str], object]):
    envelope = parse_sft_output(text)
    if envelope is None:
        return None
    parsed = parser(envelope.response)
    if parsed is None:
        return None
    return {
        "thinking": envelope.thinking,
        "response": envelope.response,
        "parsed": parsed,
    }


def build_sft_direct_prompt(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    return f"""Translate this text from {source_lang} to {target_lang} faithfully and naturally.

Source:
{source_text}"""


def build_sft_flash_gpe_candidate_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    max_candidates: int = 4,
    *,
    exact_count: bool = True,
) -> str:
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2")
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    if exact_count:
        count = f"exactly {max_candidates}"
    else:
        count = f"as many as useful, up to {max_candidates}"
    return f"""Translate this text from {source_lang} to {target_lang} and produce {count} meaningfully different complete translations. Keep every translation faithful and natural.

Source:
{source_text}"""


def build_sft_gpe_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    candidates: list[str],
) -> str:
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    rendered = "\n\n".join(
        f"Candidate {candidate_identifiers[index]}:\n{candidate}"
        for index, candidate in enumerate(candidates)
    )
    return f"""Produce the best {target_lang} translation of the {source_lang} source using the candidates. Correct errors and combine candidates only when useful.

Source:
{source_text}

{rendered}"""


def build_sft_flash_gpe_post_edit_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    candidates: list[str],
) -> str:
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    rendered = "\n\n".join(
        f"Candidate {index + 1}:\n{candidate}"
        for index, candidate in enumerate(candidates)
    )
    return f"""Produce the best {target_lang} translation of the {source_lang} source using the candidates. Correct errors and combine candidates only when useful.

Source:
{source_text}

{rendered}"""


def build_scd_stage1_prompt(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    return f"""Analyze this {source_lang} source into meaningful translation segments. For each segment, explore several plausible {target_lang} translations with different interpretations or wording. Do not produce the complete translation yet.

Source:
{source_text}"""


def build_scd_followup_prompt(source_lang: str, target_lang: str) -> str:
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    return f"""Using the source and segment candidates above, produce one faithful {target_lang} translation from {source_lang}. Resolve cross-segment dependencies, reject errors, and polish the complete translation for naturalness and consistency."""


def _require_content(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _reject_protocol_tags(value: str, name: str) -> None:
    tags = (THINKING_OPEN, THINKING_CLOSE, RESPONSE_OPEN, RESPONSE_CLOSE)
    if any(tag in value for tag in tags):
        raise ValueError(f"{name} contains a reserved SFT protocol tag")
