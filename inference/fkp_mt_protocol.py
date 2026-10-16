"""Shared simple protocol for one-pass key-point analysis and translation."""

import json
import re
from typing import Optional

from inference.run_oss_key_points import validate_key_points_result
from inference.run_oss_key_points_mt import extract_final_translation


SIMPLE_PROTOCOL_SEPARATOR = "---"
SIMPLE_PROTOCOL_CONNECTOR = "Now, use the key points to produce the complete final translation."
ANALYSIS_HEADING = "# Step-by-step Analysis"


def format_fkp_output(
    key_points_thinking: str, key_points_response: str,
    convergent_thinking: str, convergent_response: str,
) -> str:
    values = (key_points_thinking, key_points_response, convergent_thinking, convergent_response)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Both stages require non-empty thinking and responses")
    return (
        f"{ANALYSIS_HEADING}\n\n{key_points_thinking.strip()}\n\n"
        f"# Key Points\n\n{key_points_response.strip()}\n\n"
        f"{SIMPLE_PROTOCOL_SEPARATOR}\n\n{SIMPLE_PROTOCOL_CONNECTOR}\n\n"
        f"{ANALYSIS_HEADING}\n\n{convergent_thinking.strip()}\n\n"
        f"# Final Translation\n\n{convergent_response.strip()}"
    )


def _split_stage(text: str, heading: str):
    text = text.strip()
    start = re.match(rf"^{re.escape(ANALYSIS_HEADING)}[ \t]*\n", text)
    bodies = list(re.finditer(rf"(?m)^{re.escape(heading)}[ \t]*$", text))
    if start is None or len(bodies) != 1:
        return None
    body = bodies[0]
    thinking = text[start.end():body.start()].strip()
    response = text[body.end():].strip()
    if not thinking or not response:
        return None
    return thinking, response


def _parse_strict_fkp_response(text: Optional[str], source_text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    text = text.replace("\r\n", "\n").strip()
    connectors = list(re.finditer(
        rf"(?m)^[ \t]*{re.escape(SIMPLE_PROTOCOL_SEPARATOR)}[ \t]*\n{{2,}}"
        rf"[ \t]*{re.escape(SIMPLE_PROTOCOL_CONNECTOR)}[ \t]*$", text,
    ))
    if len(connectors) != 1:
        return None
    connector = connectors[0]
    before, after = text[:connector.start()], text[connector.end():]
    if not before.endswith("\n\n") or not after.startswith("\n\n"):
        return None
    first = _split_stage(before, "# Key Points")
    second = _split_stage(after, "# Final Translation")
    if first is None or second is None:
        return None
    try:
        value = json.loads(first[1])
    except (ValueError, TypeError):
        return None
    points = validate_key_points_result(value, source_text)
    translation = extract_final_translation(second[1])
    if points is None or translation is None:
        return None
    return {
        "key_points_thinking": first[0], "key_points_response": first[1],
        "key_points": points["key_points"],
        "convergent_thinking": second[0], "convergent_response": second[1],
        "parsed": translation,
    }


def parse_fkp_response(
    text: Optional[str], source_text: str, *, relaxed_parsing: bool = False,
) -> Optional[dict]:
    """Optionally accept a final translation without a valid two-stage envelope."""
    parsed = _parse_strict_fkp_response(text, source_text)
    if parsed is not None or not relaxed_parsing or not isinstance(text, str):
        return parsed
    text = text.replace("\r\n", "\n").strip()
    headings = list(re.finditer(
        r"(?im)^[ \t]*#{1,6}[ \t]+Final Translation[ \t]*:?[ \t]*$", text,
    ))
    response = text[headings[-1].end():].strip() if headings else text
    blocks = list(re.finditer(
        r"(?m)^[ \t]*```([^\n`]*)\n(.*?)\n[ \t]*```[ \t]*(?=\n|$)",
        response, re.DOTALL,
    ))
    if blocks:
        block = blocks[-1]
        translation = block.group(2).strip()
        # A key-point JSON block alone is not a final translation.
        if block.group(1).strip().lower() == "json":
            return None
        try:
            value = json.loads(translation)
        except ValueError:
            value = None
        if isinstance(value, dict) and "key_points" in value:
            return None
    elif headings and "```" not in response:
        translation = response
    else:
        return None
    if not translation:
        return None

    # Keep valid key points when possible, but never require them for acceptance.
    points_response = ""
    points = None
    key_heading = re.search(r"(?im)^#{1,6}[ \t]+Key Points[ \t]*:?[ \t]*$", text)
    if key_heading:
        points_response = text[key_heading.end():].lstrip()
        try:
            value, end = json.JSONDecoder().raw_decode(points_response)
            points_response = points_response[:end]
            points = validate_key_points_result(value, source_text)
        except (ValueError, TypeError):
            points_response = ""
    return {
        "key_points_thinking": "", "key_points_response": points_response,
        "key_points": points["key_points"] if points is not None else [],
        "convergent_thinking": "", "convergent_response": response,
        "parsed": translation,
    }
