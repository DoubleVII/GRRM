"""Deprecated Fused FlashGPE inference for checkpoints trained on JSON output."""

from typing import Optional, Union

from inference.run_mt import _block_extractor, load_model_tokenizer
from inference.sft_mt_protocol import parse_fused_task_output
from inference.legacy.oss_flash_gpe_json import extract_candidate_response


def _validate_prompt_type(prompt_type: str, max_candidates: int) -> None:
    if prompt_type not in {"adaptive", "fixed_4", "fixed_16"}:
        raise ValueError("legacy prompt_type must be adaptive, fixed_4, or fixed_16")
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2")
    if prompt_type == "fixed_4" and max_candidates != 4:
        raise ValueError("fixed_4 prompt_type requires max_candidates=4")
    if prompt_type == "fixed_16" and max_candidates != 16:
        raise ValueError("fixed_16 prompt_type requires max_candidates=16")


def _build_prompt(source_lang: str, target_lang: str, source_text: str, max_candidates: int, prompt_type: str) -> str:
    exact = prompt_type in {"fixed_4", "fixed_16"}
    count = f"exactly {max_candidates}" if exact else f"as many as useful, up to {max_candidates}"
    return f"""Translate this text from {source_lang} to {target_lang}. First produce {count} meaningfully different complete translations, keeping each one faithful and natural. Then review those candidates, correct their errors, and produce the best final translation.

Source:
{source_text}"""


def _parse_ffgpe_response(text: str, max_candidates: int, prompt_type: str) -> Optional[dict]:
    exact_count = prompt_type in {"fixed_4", "fixed_16"}
    return parse_fused_task_output(
        text,
        lambda response: extract_candidate_response(
            response, max_candidates, exact_count=exact_count
        ),
        _block_extractor,
    )


def func_call(
    model_path: str,
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    max_candidates: int = 4,
    prompt_type: str = "fixed_4",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 8192,
    retry: int = 3,
    use_chat_template: bool = True,
    model=None,
    tokenizer=None,
    **kwargs,
):
    from vllm import SamplingParams

    _validate_prompt_type(prompt_type, max_candidates)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)
    if len(src_list) != len(src_langs) or len(src_list) != len(trg_langs):
        raise ValueError("src_list, src_langs, and trg_langs must have the same length")
    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path, **kwargs)

    prompts = []
    for source, source_lang, target_lang in zip(src_list, src_langs, trg_langs):
        prompt = _build_prompt(source_lang, target_lang, source, max_candidates, prompt_type)
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)

    raw_outputs: list[Optional[str]] = [None] * len(src_list)
    parsed_outputs: list[Optional[dict]] = [None] * len(src_list)
    for attempt in range(retry + 1):
        indices = [i for i, value in enumerate(parsed_outputs) if value is None]
        if not indices:
            break
        params = SamplingParams(
            temperature=min(1.0, temperature + 0.1 * attempt),
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=1,
        )
        outputs = model.generate([prompts[i] for i in indices], params)
        for index, output in zip(indices, outputs):
            if not output.outputs:
                continue
            raw_outputs[index] = output.outputs[0].text
            parsed_outputs[index] = _parse_ffgpe_response(
                raw_outputs[index], max_candidates, prompt_type
            )

    candidates = [value["candidates"] if value else [] for value in parsed_outputs]
    return {
        "responses": [value["parsed"] if value else "Translation Failed." for value in parsed_outputs],
        "raw_outputs": raw_outputs,
        "parsed_outputs": parsed_outputs,
        "candidates": candidates,
        "candidate_counts": [len(value) for value in candidates],
        "parser_valid": [value is not None for value in parsed_outputs],
    }


if __name__ == "__main__":
    import fire
    fire.Fire(func_call)
