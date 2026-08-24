"""Deprecated JSON FlashGPE protocol.

New data must use the Markdown protocol in ``oss_flash_gpe_prompts``.
"""

import json
from typing import Optional


def extract_candidate_response(
    response: str, max_candidates: int, *, exact_count: bool = False
) -> Optional[list[str]]:
    if not isinstance(response, str):
        return None
    try:
        value = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        text = response.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[:-3].strip()
        start = text.find("{")
        if start == -1:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except (TypeError, json.JSONDecodeError):
            return None
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
