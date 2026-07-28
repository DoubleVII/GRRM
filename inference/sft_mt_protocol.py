import re
from dataclasses import dataclass
from typing import Callable, Optional


THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"
RESPONSE_OPEN = "<response>"
RESPONSE_CLOSE = "</response>"

OUTPUT_INSTRUCTION = f"""Return the complete assistant output in exactly this format:
{THINKING_OPEN}
[your reasoning]
{THINKING_CLOSE}
{RESPONSE_OPEN}
[the task response in the format requested above]
{RESPONSE_CLOSE}

Both sections must be non-empty. Do not output anything outside these tags."""


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


def add_output_instruction(prompt: str) -> str:
    prompt = _require_content(prompt, "prompt")
    return f"{prompt.rstrip()}\n\n{OUTPUT_INSTRUCTION}"


def build_scd_followup_prompt(source_lang: str, target_lang: str) -> str:
    return add_output_instruction(
        f"""Now perform the convergent stage of the translation from {source_lang} to {target_lang}.

Use the complete source in the first user message and the segment analysis and candidates in your previous response. Treat those candidates as exploratory evidence, not as authority. Reject or repair any candidate that conflicts with the source, then select, edit, or combine the strongest choices into one faithful and coherent complete translation.

Perform a mandatory whole-text polish pass for idiomatic phrasing, sentence structure, cohesion, punctuation, and consistent register. Recheck that polishing has not changed facts, entities, numbers, negation, logical relationships, emphasis, tone, or formatting.

Inside the response section, output only the final translation between the exact tags below, with no other text:
<final_translation>...</final_translation>"""
    )


def _require_content(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _reject_protocol_tags(value: str, name: str) -> None:
    tags = (THINKING_OPEN, THINKING_CLOSE, RESPONSE_OPEN, RESPONSE_CLOSE)
    if any(tag in value for tag in tags):
        raise ValueError(f"{name} contains a reserved SFT protocol tag")
