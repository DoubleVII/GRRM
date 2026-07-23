from typing import List, Optional, Union

from inference.prompts import Task_format, get_GQMPE_prompt
from inference.run_mt import _block_extractor
from inference.run_rm_SQM import load_model_tokenizer
from utils.config import candidate_identifiers
from utils.helpers import _ranking_to_scores, _score_to_rank, parse_score_text


POST_EDIT_ANALYSIS_HEADER = "# Post-edit Analysis"
FINAL_TRANSLATION_HEADER = "# Final post-edited translation"
FINAL_RANKING_HEADER = "### Final Ranking:"
SCORES_HEADER = "### Scores:"


def _single_nonempty_line(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if len(lines) == 1 else None


def _validate_ranking(ranking_text: str, expected_num: int) -> bool:
    try:
        if "<" in ranking_text:
            return False
        groups = ranking_text.split(">")
        identifiers = [
            identifier.strip()
            for group in groups
            for identifier in group.split("=")
        ]
        expected = candidate_identifiers[:expected_num]
        return len(identifiers) == expected_num and sorted(identifiers) == sorted(expected)
    except Exception:
        return False


def _ranking_tiers(ranking_text: str) -> list[set[str]]:
    return [
        {identifier.strip() for identifier in group.split("=")}
        for group in ranking_text.split(">")
    ]


def _parse_scores(score_text: str, expected_num: int) -> Optional[list[int]]:
    score_dict = parse_score_text(score_text)
    expected_identifiers = candidate_identifiers[:expected_num]
    if score_dict is None or set(score_dict) != set(expected_identifiers):
        return None
    return [score_dict[identifier] for identifier in expected_identifiers]


def extract_response(
    output_text: str,
    prompt_type: str,
    expected_score_num: int,
) -> Optional[dict]:
    if prompt_type not in Task_format:
        raise ValueError(f"Invalid prompt_type: {prompt_type}")

    output_text = output_text.strip()
    post_edit_analysis_index = output_text.rfind(POST_EDIT_ANALYSIS_HEADER)
    final_translation_index = output_text.rfind(FINAL_TRANSLATION_HEADER)
    if (
        post_edit_analysis_index == -1
        or final_translation_index == -1
        or final_translation_index <= post_edit_analysis_index
    ):
        return None

    post_edit_analysis = output_text[
        post_edit_analysis_index
        + len(POST_EDIT_ANALYSIS_HEADER):final_translation_index
    ].strip()
    post_edit_mt = _block_extractor(output_text)
    if not post_edit_analysis or post_edit_mt is None or not post_edit_mt.strip():
        return None

    gqm_text = output_text[:post_edit_analysis_index].strip()
    ranking_text = None
    score_text = None

    if prompt_type == "score":
        lines = gqm_text.splitlines()
        score_line_index = next(
            (i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()),
            None,
        )
        if score_line_index is None:
            return None
        score_text = lines[score_line_index].strip()
        gqm_analysis = "\n".join(lines[:score_line_index]).strip()
        scores = _parse_scores(score_text, expected_score_num)
    elif prompt_type == "ranking":
        ranking_header_index = gqm_text.rfind(FINAL_RANKING_HEADER)
        if ranking_header_index == -1:
            return None
        gqm_analysis = gqm_text[:ranking_header_index].strip()
        ranking_text = _single_nonempty_line(
            gqm_text[ranking_header_index + len(FINAL_RANKING_HEADER):]
        )
        if ranking_text is None or not _validate_ranking(
            ranking_text, expected_score_num
        ):
            return None
        score_dict = _ranking_to_scores(ranking_text)
        scores = [
            score_dict[identifier]
            for identifier in candidate_identifiers[:expected_score_num]
        ]
    else:
        ranking_header_index = gqm_text.rfind(FINAL_RANKING_HEADER)
        score_header_index = gqm_text.rfind(SCORES_HEADER)
        if (
            ranking_header_index == -1
            or score_header_index == -1
            or score_header_index <= ranking_header_index
        ):
            return None
        gqm_analysis = gqm_text[:ranking_header_index].strip()
        ranking_text = _single_nonempty_line(
            gqm_text[
                ranking_header_index + len(FINAL_RANKING_HEADER):score_header_index
            ]
        )
        score_text = _single_nonempty_line(
            gqm_text[score_header_index + len(SCORES_HEADER):]
        )
        if (
            ranking_text is None
            or score_text is None
            or not _validate_ranking(ranking_text, expected_score_num)
        ):
            return None
        scores = _parse_scores(score_text, expected_score_num)
        if scores is not None:
            score_dict = {
                candidate_identifiers[i]: score for i, score in enumerate(scores)
            }
            if _ranking_tiers(ranking_text) != _ranking_tiers(
                _score_to_rank(score_dict)
            ):
                return None

    if not gqm_analysis or scores is None:
        return None
    return {
        "gqm_analysis": gqm_analysis,
        "ranking_text": ranking_text,
        "score_text": score_text,
        "scores": scores,
        "post_edit_analysis": post_edit_analysis,
        "post_edit_mt": post_edit_mt.strip(),
    }


def _generate(model, prompt_list: list[str], sampling_params):
    outputs = model.generate(prompt_list, sampling_params)
    return [output.outputs[0].text for output in outputs]


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
    model=None,
    tokenizer=None,
):
    from vllm import SamplingParams

    if prompt_type not in Task_format:
        raise ValueError(f"Invalid prompt_type: {prompt_type}")
    item_count = len(src_list)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * item_count
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * item_count
    if not (
        item_count == len(mt_list) == len(src_langs) == len(trg_langs)
    ):
        raise ValueError(
            "src_list, mt_list, src_langs, and trg_langs must have the same length."
        )

    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    prompt_list = []
    for src_text, mt_texts, src_lang, trg_lang in zip(
        src_list, mt_list, src_langs, trg_langs
    ):
        prompt = get_GQMPE_prompt(
            src_lang,
            trg_lang,
            src_text,
            mt_texts,
            prompt_format=prompt_type,
        )
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_list.append(input_text)

    responses = _generate(model, prompt_list, sampling_params)
    parsed = [
        extract_response(response, prompt_type, len(mt_texts))
        for response, mt_texts in zip(responses, mt_list)
    ]

    for retry_count in range(retry):
        failed_indices = [i for i, item in enumerate(parsed) if item is None]
        if not failed_indices:
            break
        print(
            f"Retry attempt {retry_count + 1}: "
            f"{len(failed_indices)} failed items remaining..."
        )
        retry_params = SamplingParams(
            temperature=1.0,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        retry_responses = _generate(
            model,
            [prompt_list[i] for i in failed_indices],
            retry_params,
        )
        for index, response in zip(failed_indices, retry_responses):
            parsed_item = extract_response(
                response, prompt_type, len(mt_list[index])
            )
            responses[index] = response
            parsed[index] = parsed_item

    failed_indices = [i for i, item in enumerate(parsed) if item is None]
    if failed_indices:
        print(
            f"Warning: {len(failed_indices)} items still failed after {retry} retries."
        )

    return {
        "scores": [item["scores"] if item is not None else None for item in parsed],
        "post_edit_mt": [
            item["post_edit_mt"] if item is not None else "Translation Failed."
            for item in parsed
        ],
        "responses": responses,
        "parsed": parsed,
    }


if __name__ == "__main__":
    pass
