from typing import List, Optional, Union

from inference.prompts import Task_format, get_GQM_GPE_prompt, get_GQM_prompt
from inference.run_mt import _block_extractor, load_model_tokenizer
from inference.run_rm_GQM import extract_score


def _as_list(value: Union[str, List[str]], n: int) -> List[str]:
    if isinstance(value, str):
        return [value] * n
    return value


def _extract_gqm_analysis(response: Optional[str], prompt_type: str) -> Optional[str]:
    if response is None:
        return None

    response = response.strip()
    if not response:
        return None

    if prompt_type == "ranking_score":
        marker = "### Scores:"
        marker_index = response.rfind(marker)
        if marker_index != -1:
            return response[:marker_index].strip()
    elif prompt_type == "ranking":
        marker = "### Final Ranking:"
        marker_index = response.rfind(marker)
        if marker_index != -1:
            return response[:marker_index].strip()

    lines = response.splitlines()
    if len(lines) <= 1:
        return None
    return "\n".join(lines[:-1]).strip()


def _extract_post_edit(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None
    post_edit = _block_extractor(response)
    if post_edit is None:
        return None
    post_edit = post_edit.strip()
    if not post_edit:
        return None
    return post_edit


def _generate_texts(model, prompts: list[str], sampling_params) -> list[str]:
    outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def _run_gqm_round(
    model,
    prompt_list: list[str],
    mt_list: list[list[str]],
    prompt_type: str,
    sampling_params,
    retry: int,
    retry_temperature: float,
):
    from vllm import SamplingParams

    response_list = _generate_texts(model, prompt_list, sampling_params)
    score_list = [
        extract_score(response, prompt_type, len(mt_texts))
        for response, mt_texts in zip(response_list, mt_list)
    ]

    failed_indices = [i for i, scores in enumerate(score_list) if scores is None]
    retry_count = 0

    while failed_indices and retry_count < retry:
        retry_count += 1
        print(f"GQM retry attempt {retry_count}: {len(failed_indices)} failed items remaining...")

        retry_prompts = [prompt_list[i] for i in failed_indices]
        retry_mt_list = [mt_list[i] for i in failed_indices]
        retry_sampling_params = SamplingParams(
            temperature=retry_temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
        )
        retry_responses = _generate_texts(model, retry_prompts, retry_sampling_params)
        retry_scores = [
            extract_score(response, prompt_type, len(mt_texts))
            for response, mt_texts in zip(retry_responses, retry_mt_list)
        ]

        for idx, response, scores in zip(failed_indices, retry_responses, retry_scores):
            response_list[idx] = response
            score_list[idx] = scores

        failed_indices = [i for i, scores in enumerate(score_list) if scores is None]

    if failed_indices:
        print(f"Warning: {len(failed_indices)} GQM items still failed after {retry} retries.")

    return response_list, score_list


def _run_gpe_round(
    model,
    prompt_list: list[str],
    sampling_params,
    retry: int,
    retry_temperature: float,
):
    from vllm import SamplingParams

    response_list = _generate_texts(model, prompt_list, sampling_params)
    post_edit_list = [_extract_post_edit(response) for response in response_list]

    failed_indices = [i for i, post_edit in enumerate(post_edit_list) if post_edit is None]
    retry_count = 0

    while failed_indices and retry_count < retry:
        retry_count += 1
        print(f"GPE retry attempt {retry_count}: {len(failed_indices)} failed items remaining...")

        retry_prompts = [prompt_list[i] for i in failed_indices]
        retry_sampling_params = SamplingParams(
            temperature=retry_temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
        )
        retry_responses = _generate_texts(model, retry_prompts, retry_sampling_params)
        retry_post_edits = [_extract_post_edit(response) for response in retry_responses]

        for idx, response, post_edit in zip(failed_indices, retry_responses, retry_post_edits):
            response_list[idx] = response
            post_edit_list[idx] = post_edit

        failed_indices = [i for i, post_edit in enumerate(post_edit_list) if post_edit is None]

    if failed_indices:
        print(f"Warning: {len(failed_indices)} GPE items still failed after {retry} retries.")

    post_edit_list = [
        post_edit if post_edit is not None else "Translation Failed."
        for post_edit in post_edit_list
    ]
    return response_list, post_edit_list


def func_call(
    model_path: str,
    src_list: list[str],
    mt_list: list[list[str]],
    src_langs: Union[str, List[str]],
    trg_langs: Union[str, List[str]],
    temperature: float = 0.4,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 6,
    prompt_type: str = "ranking_score",
    add_example: bool = False,
    retry_temperature: float = 1.0,
    model=None,
    tokenizer=None,
    notes_list: Optional[list[Optional[str]]] = None,
    ref_text_list: Optional[list[Optional[str]]] = None,
    ref_langs: Optional[Union[str, List[Optional[str]]]] = None,
):
    from vllm import SamplingParams

    if prompt_type not in Task_format:
        raise ValueError(f"prompt_type must be one of {Task_format.keys()}")

    n = len(src_list)
    src_langs = _as_list(src_langs, n)
    trg_langs = _as_list(trg_langs, n)

    if notes_list is None:
        notes_list = [None] * n
    if ref_text_list is None:
        ref_text_list = [None] * n
    if ref_langs is None:
        ref_langs = [None] * n
    elif isinstance(ref_langs, str):
        ref_langs = [ref_langs] * n

    if not (
        len(src_list)
        == len(mt_list)
        == len(src_langs)
        == len(trg_langs)
        == len(notes_list)
        == len(ref_text_list)
        == len(ref_langs)
    ):
        raise ValueError(
            "src_list, mt_list, src_langs, trg_langs, notes_list, ref_text_list, "
            "and ref_langs must have the same length."
        )

    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    gqm_messages_list = []
    gqm_prompt_list = []
    gqm_user_prompts = []
    for src_text, mt_texts, src_lang, trg_lang, notes, ref_text, ref_lang in zip(
        src_list, mt_list, src_langs, trg_langs, notes_list, ref_text_list, ref_langs
    ):
        prompt = get_GQM_prompt(
            src_lang,
            trg_lang,
            src_text,
            mt_texts,
            prompt_type,
            add_example=add_example,
            notes=notes,
            ref_text=ref_text,
            ref_lang=ref_lang,
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        gqm_messages_list.append(messages)
        gqm_prompt_list.append(input_text)
        gqm_user_prompts.append(prompt)

    gqm_response_list, gqm_score_list = _run_gqm_round(
        model,
        gqm_prompt_list,
        mt_list,
        prompt_type,
        sampling_params,
        retry,
        retry_temperature,
    )

    gpe_prompt = get_GQM_GPE_prompt()
    gpe_prompt_list = []
    gpe_messages_list = []
    for messages, gqm_response in zip(gqm_messages_list, gqm_response_list):
        cascade_messages = [
            messages[0],
            {"role": "assistant", "content": gqm_response},
            {"role": "user", "content": gpe_prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            cascade_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        gpe_messages_list.append(cascade_messages)
        gpe_prompt_list.append(input_text)

    gpe_response_list, post_edit_list = _run_gpe_round(
        model,
        gpe_prompt_list,
        sampling_params,
        retry,
        retry_temperature,
    )

    gqm_analysis_list = [
        _extract_gqm_analysis(response, prompt_type)
        for response in gqm_response_list
    ]
    gqm_parsed_list = [
        {"scores": scores, "analysis": analysis}
        for scores, analysis in zip(gqm_score_list, gqm_analysis_list)
    ]

    return {
        "post_edit_mt": post_edit_list,
        "responses": gpe_response_list,
        "gpe_responses": gpe_response_list,
        "gqm_responses": gqm_response_list,
        "gqm_scores": gqm_score_list,
        "gqm_analysis": gqm_analysis_list,
        "gqm_parsed": gqm_parsed_list,
        "gqm_prompts": gqm_user_prompts,
        "gpe_messages": gpe_messages_list,
    }


if __name__ == "__main__":
    pass
