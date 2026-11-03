"""Explicit response envelopes; annotation validation remains unchanged."""
import re

from utils.hard_protocol import parse_json


def parse_prediction_response(text, response_format="json"):
    if response_format == "json":
        return parse_json(text)
    if response_format != "analysis_codeblock":
        raise ValueError("Unknown prediction response format")
    # Require nonempty analysis and one final, complete JSON fence. Never scan
    # arbitrary prose for braces or repair model-generated JSON/field values.
    match = re.fullmatch(
        r"\s*# Step-by-step Analysis\s*\n(.+?)\n# Final JSON\s*\n"
        r"```json[ \t]*\n(.*?)\n```\s*", text or "", re.DOTALL)
    if not match or not match[1].strip() or "```" in match[1]:
        return None
    import json
    try:
        value = json.loads(match[2])
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None
