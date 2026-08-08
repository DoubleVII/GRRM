from typing import Optional, Union

from inference.prompts import build_ffgpe_prompt, validate_ffgpe_prompt_type
from inference.run_mt import _block_extractor, load_model_tokenizer
from inference.run_oss_flash_gpe_mt import extract_candidate_response
from inference.sft_mt_protocol import parse_fused_task_output


def _parse_ffgpe_response(
    text: str,
    max_candidates: int,
    prompt_type: str,
) -> Optional[dict]:
    exact_count = prompt_type in {"fixed_4", "fixed_16"}
    return parse_fused_task_output(
        text,
        lambda response: extract_candidate_response(
            response,
            max_candidates,
            exact_count=exact_count,
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
    """Run one-pass FFGPE generation and return final translations."""
    from vllm import SamplingParams

    validate_ffgpe_prompt_type(prompt_type, max_candidates)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)
    if len(src_list) != len(src_langs) or len(src_list) != len(trg_langs):
        raise ValueError("src_list, src_langs, and trg_langs must have the same length")
    if retry < 0:
        raise ValueError("retry must be non-negative")

    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path, **kwargs)

    prompt_list = []
    for source, source_lang, target_lang in zip(
        src_list, src_langs, trg_langs
    ):
        prompt = build_ffgpe_prompt(
            source_lang,
            target_lang,
            source,
            max_candidates=max_candidates,
            prompt_type=prompt_type,
        )
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_list.append(prompt)

    raw_outputs: list[Optional[str]] = [None] * len(src_list)
    parsed_outputs: list[Optional[dict]] = [None] * len(src_list)
    for attempt in range(retry + 1):
        run_indices = [
            index
            for index, parsed in enumerate(parsed_outputs)
            if parsed is None
        ]
        if not run_indices:
            break
        sampling_params = SamplingParams(
            temperature=min(1.0, temperature + 0.1 * attempt),
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=1,
        )
        outputs = model.generate(
            [prompt_list[index] for index in run_indices],
            sampling_params,
        )
        for index, output in zip(run_indices, outputs):
            if not output.outputs:
                continue
            raw_text = output.outputs[0].text
            parsed = _parse_ffgpe_response(
                raw_text, max_candidates, prompt_type
            )
            raw_outputs[index] = raw_text
            parsed_outputs[index] = parsed

    responses = [
        parsed["parsed"] if parsed is not None else "Translation Failed."
        for parsed in parsed_outputs
    ]
    candidates = [
        parsed["candidates"] if parsed is not None else []
        for parsed in parsed_outputs
    ]
    return {
        "responses": responses,
        "raw_outputs": raw_outputs,
        "parsed_outputs": parsed_outputs,
        "candidates": candidates,
        "candidate_counts": [len(values) for values in candidates],
        "parser_valid": [parsed is not None for parsed in parsed_outputs],
    }


if __name__ == "__main__":
    import fire

    fire.Fire(func_call)
