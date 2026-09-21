from typing import Optional, Union

from inference.ffgpe_protocol import validate_ffgpe_protocol
from inference.prompts import build_ffgqm_prompt
from inference.run_inst_flash_gpe_mt import extract_candidate_response
from inference.run_inst_flash_gqm_mt import extract_gqm_response
from inference.run_mt import load_model_tokenizer
from inference.sft_mt_protocol import parse_simple_gqm_task_output


def parse_ffgqm_response(
    text: str,
    max_candidates: int,
) -> Optional[dict]:
    candidate_parser = lambda response: extract_candidate_response(
        response, max_candidates, explicit_analysis=False
    )
    gqm_parser = lambda response: extract_gqm_response(
        response, max_candidates, explicit_analysis=False
    )
    return parse_simple_gqm_task_output(text, candidate_parser, gqm_parser)


def _parse_ffgqm_response(text: str, max_candidates: int) -> Optional[dict]:
    return parse_ffgqm_response(text, max_candidates)


def func_call(
    model_path: str,
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    max_candidates: int = 4,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 8192,
    retry: int = 3,
    protocol: str = "simple",
    use_chat_template: bool = True,
    model=None,
    tokenizer=None,
    **kwargs,
):
    """Run one-pass fused FlashGQM generation with the simple protocol."""
    if protocol != "simple":
        raise ValueError("FFGQM only supports protocol='simple'")
    validate_ffgpe_protocol(protocol)
    if not 2 <= max_candidates <= 8:
        raise ValueError("max_candidates must be between 2 and 8")
    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)
    if len(src_list) != len(src_langs) or len(src_list) != len(trg_langs):
        raise ValueError("src_list, src_langs, and trg_langs must have the same length")
    if retry < 0:
        raise ValueError("retry must be non-negative")

    from vllm import SamplingParams

    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path, **kwargs)

    prompt_list = []
    for source, source_lang, target_lang in zip(src_list, src_langs, trg_langs):
        prompt = build_ffgqm_prompt(
            source_lang,
            target_lang,
            source,
            max_candidates=max_candidates,
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
            index for index, parsed in enumerate(parsed_outputs) if parsed is None
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
            [prompt_list[index] for index in run_indices], sampling_params
        )
        for index, output in zip(run_indices, outputs):
            if not output.outputs:
                continue
            raw_text = output.outputs[0].text
            raw_outputs[index] = raw_text
            parsed_outputs[index] = parse_ffgqm_response(raw_text, max_candidates)

    responses = []
    candidates = []
    scores = []
    selected_indices = []
    parser_valid = []
    for parsed in parsed_outputs:
        if parsed is None:
            responses.append("Translation Failed.")
            candidates.append([])
            scores.append(None)
            selected_indices.append(None)
            parser_valid.append(False)
            continue
        values = parsed["candidates"]
        item_scores = parsed["gqm"]["scores"]
        selected = max(range(len(item_scores)), key=item_scores.__getitem__)
        responses.append(values[selected])
        candidates.append(values)
        scores.append(item_scores)
        selected_indices.append(selected)
        parser_valid.append(True)
    return {
        "responses": responses,
        "raw_outputs": raw_outputs,
        "parsed_outputs": parsed_outputs,
        "candidates": candidates,
        "scores": scores,
        "selected_candidate_indices": selected_indices,
        "candidate_counts": [len(values) for values in candidates],
        "parser_valid": parser_valid,
    }


if __name__ == "__main__":
    import fire

    fire.Fire(func_call)
