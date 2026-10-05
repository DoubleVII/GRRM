from typing import Optional, Union

from inference.oss_key_points_prompts import build_key_points_prompt
from inference.run_oss_SQM import init_oss_model, load_encoding
from inference.run_oss_diverse_mt import (
    _generate_with_retries,
    _prepare_inputs,
    extract_json_object,
)


def validate_key_points_result(value: Optional[dict], source_text: str) -> Optional[dict]:
    if not isinstance(value, dict) or set(value) != {"key_points"}:
        return None
    points = value["key_points"]
    if not isinstance(points, list):
        return None
    for point in points:
        if not isinstance(point, dict) or set(point) != {
            "source_span", "explanation", "translations"
        }:
            return None
        span = point["source_span"]
        explanation = point["explanation"]
        translations = point["translations"]
        if not isinstance(span, str) or not span.strip() or span not in source_text:
            return None
        if not isinstance(explanation, str) or not explanation.strip():
            return None
        if not isinstance(translations, list) or not translations:
            return None
        if any(not isinstance(item, str) or not item.strip() for item in translations):
            return None
    return value


def parse_key_points_response(response: Optional[str], source_text: str):
    if not isinstance(response, str) or not response.strip():
        return None
    return validate_key_points_result(extract_json_object(response), source_text)


def run_key_points_stage(
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    model=None,
    model_path: str = "openai/gpt-oss-120b",
    reasoning_effort: Optional[str] = "medium",
    temperature: float = 0.3,
    top_p: float = 0.8,
    max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    n = len(src_list)
    src_langs = [src_langs] * n if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * n if isinstance(trg_langs, str) else trg_langs
    if not (n == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")
    llm = init_oss_model(model_path) if model is None else model
    encoding = load_encoding()
    prompts = [
        build_key_points_prompt(sl, tl, source)
        for source, sl, tl in zip(src_list, src_langs, trg_langs)
    ]
    parsers = [lambda text, source=source: parse_key_points_response(text, source)
               for source in src_list]
    # The shared retry helper supports an item-aware parser, preserving each source
    # text while retrying only failed generations.
    results = _generate_with_retries(
        llm,
        _prepare_inputs(prompts, encoding, reasoning_effort),
        lambda text, index: parsers[index](text),
        encoding,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        retry=retry,
        parser_with_index=True,
    )
    return {
        "key_points": [result["parsed"] for result in results],
        "responses": [result["response"] for result in results],
        "thinking": [result["thinking"] for result in results],
        "prompts": prompts,
    }
