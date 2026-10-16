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


def parse_fkp_response(text: Optional[str], source_text: str) -> Optional[dict]:
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
