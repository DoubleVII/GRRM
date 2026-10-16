"""Generate key points and a final translation in one local-model completion."""

from typing import Union

from inference.fkp_mt_prompts import build_fkp_mt_prompt
from inference.fkp_mt_protocol import parse_fkp_response
from inference.run_mt import load_model_tokenizer


def func_call(
    model_path: str,
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 8192,
    retry: int = 3,
    relaxed_parsing: bool = False,
    use_chat_template: bool = True,
    model=None,
    tokenizer=None,
    **kwargs,
) -> dict:
    n = len(src_list)
    src_langs = [src_langs] * n if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * n if isinstance(trg_langs, str) else trg_langs
    if not (n == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")
    if retry < 0:
        raise ValueError("retry must be non-negative")
    prompts = [build_fkp_mt_prompt(sl, tl, source)
               for source, sl, tl in zip(src_list, src_langs, trg_langs)]
    raw_outputs = [None] * n
    parsed_outputs = [None] * n
    if n:
        from vllm import SamplingParams

        if model is None or (use_chat_template and tokenizer is None):
            model, tokenizer = load_model_tokenizer(model_path, **kwargs)
        if use_chat_template:
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) for prompt in prompts]
        # Explicit analysis headings carry both stages' reasoning; disable a
        # separate tokenizer-native thinking envelope to match the SFT messages.
        for attempt in range(retry + 1):
            indices = [i for i, parsed in enumerate(parsed_outputs) if parsed is None]
            if not indices:
                break
            params = SamplingParams(temperature=min(1.0, temperature + 0.1 * attempt),
                                    top_p=top_p, max_tokens=max_new_tokens, n=1)
            outputs = model.generate([prompts[i] for i in indices], params)
            for i, output in zip(indices, outputs):
                if not output.outputs:
                    continue
                raw_outputs[i] = output.outputs[0].text
                parsed_outputs[i] = parse_fkp_response(
                    raw_outputs[i], src_list[i], relaxed_parsing=relaxed_parsing,
                )
    key_points = [parsed["key_points"] if parsed is not None else [] for parsed in parsed_outputs]
    return {
        "responses": [parsed["parsed"] if parsed is not None else "Translation Failed."
                      for parsed in parsed_outputs],
        "raw_outputs": raw_outputs, "parsed_outputs": parsed_outputs,
        "key_points": key_points, "key_point_counts": [len(points) for points in key_points],
        "parser_valid": [parsed is not None for parsed in parsed_outputs],
    }


if __name__ == "__main__":
    import fire

    fire.Fire(func_call)
