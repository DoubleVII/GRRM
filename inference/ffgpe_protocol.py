import re
from dataclasses import dataclass
from typing import Callable, Optional


FFGPE_PROTOCOLS = {"tag", "simple"}
SIMPLE_PROTOCOL_CONNECTOR = (
    "Now, review the candidates and produce the best final translation."
)
SIMPLE_PROTOCOL_SEPARATOR = "---"
_ANALYSIS_HEADING = "# Step-by-step Analysis"


@dataclass(frozen=True)
class SimpleFfgpeOutput:
    candidate_thinking: str
    candidate_response: str
    post_edit_thinking: str
    post_edit_response: str


def validate_ffgpe_protocol(protocol: str) -> None:
    if not isinstance(protocol, str) or protocol not in FFGPE_PROTOCOLS:
        raise ValueError("protocol must be one of: tag, simple")


def format_simple_ffgpe_output(
    candidate_response: str,
    post_edit_response: str,
) -> str:
    responses = (candidate_response, post_edit_response)
    if any(not isinstance(value, str) or not value.strip() for value in responses):
        raise ValueError("candidate and post-edit responses must be non-empty strings")
    return (
        f"{candidate_response.strip()}\n\n"
        f"{SIMPLE_PROTOCOL_SEPARATOR}\n\n"
        f"{SIMPLE_PROTOCOL_CONNECTOR}\n\n"
        f"{post_edit_response.strip()}"
    )


def parse_simple_ffgpe_output(
    text: Optional[str],
) -> Optional[SimpleFfgpeOutput]:
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n")
    connector_matches = list(re.finditer(
        rf"(?m)^[ \t]*{re.escape(SIMPLE_PROTOCOL_SEPARATOR)}[ \t]*\n{{2,}}"
        rf"[ \t]*{re.escape(SIMPLE_PROTOCOL_CONNECTOR)}[ \t]*$",
        text,
    ))
    if len(connector_matches) != 1:
        return None

    connector_match = connector_matches[0]
    before = text[:connector_match.start()]
    after = text[connector_match.end():]
    if not before.endswith("\n\n") or not after.startswith("\n\n"):
        return None
    candidate_sections = _split_analysis_section(
        before,
        r"^# Candidate 1[ \t]*$",
    )
    post_edit_sections = _split_analysis_section(
        after,
        r"^# Final Translation[ \t]*$",
    )
    if candidate_sections is None or post_edit_sections is None:
        return None
    return SimpleFfgpeOutput(*candidate_sections, *post_edit_sections)


def parse_simple_task_output(
    text: Optional[str],
    candidate_parser: Callable[[str], object],
    post_edit_parser: Callable[[str], object],
):
    envelope = parse_simple_ffgpe_output(text)
    if envelope is None:
        return None
    candidates = candidate_parser(envelope.candidate_response)
    translation = post_edit_parser(envelope.post_edit_response)
    if candidates is None or translation is None:
        return None
    return {
        "candidate_thinking": envelope.candidate_thinking,
        "candidate_response": envelope.candidate_response,
        "candidates": candidates,
        "post_edit_thinking": envelope.post_edit_thinking,
        "post_edit_response": envelope.post_edit_response,
        "parsed": translation,
    }


def _split_analysis_section(
    text: str,
    body_heading_pattern: str,
) -> Optional[tuple[str, str]]:
    text = text.strip()
    analysis_match = re.match(
        rf"^{re.escape(_ANALYSIS_HEADING)}[ \t]*(?:\r?\n)",
        text,
    )
    if analysis_match is None:
        return None
    body_match = re.search(body_heading_pattern, text, re.MULTILINE)
    if body_match is None:
        return None
    analysis = text[analysis_match.end():body_match.start()].strip()
    response = text[body_match.start():].strip()
    if not analysis or not response:
        return None
    return analysis, response
