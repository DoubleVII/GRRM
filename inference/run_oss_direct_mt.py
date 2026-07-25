from typing import Optional, Union

from inference.oss_diverse_mt_prompts import build_direct_prompt
from inference.run_oss_SQM import init_oss_model, load_encoding
from inference.run_oss_diverse_mt import (
    _generate_with_retries,
    _prepare_inputs,
    extract_final_translation,
)


def run_direct_stage(
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
        build_direct_prompt(src_lang, trg_lang, source)
        for source, src_lang, trg_lang in zip(src_list, src_langs, trg_langs)
    ]
    results = _generate_with_retries(
        llm,
        _prepare_inputs(prompts, encoding, reasoning_effort),
        extract_final_translation,
        encoding,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        retry=retry,
    )
    return {
        "translations": [result["parsed"] for result in results],
        "responses": [result["response"] for result in results],
        "thinking": [result["thinking"] for result in results],
    }
