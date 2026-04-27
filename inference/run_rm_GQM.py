from typing import Union, List, Optional
from utils.config import candidate_identifiers
from utils.helpers import parse_score_text, _ranking_to_scores
from inference.run_rm_SQM import load_model_tokenizer
from inference.prompts import get_GQM_with_notes_prompt, Task_format


def _validate_ranking(test_str: str, expected_num: int) -> bool:
    def parse_order(order_str):
        tiers = []
        for group in order_str.split('>'):
            tier = set(x.strip() for x in group.split('='))
            tiers.append(tier)
        return tiers
    
    try:
        if "<" in test_str:
            return False
        test_tiers = parse_order(test_str)
        test_count = sum(len(tiers) for tiers in test_tiers)
        if test_count != expected_num:
            return False
        for cand_id in candidate_identifiers[:expected_num]:
            if test_str.count(cand_id) != 1:
                return False
        return True
    except Exception:
        return False



def extract_score(output_text: str, prompt_type: str, expected_score_num: int) -> Optional[int]:
    output_text = output_text.strip()
    try:
        if "\n" not in output_text: # for no cot case
            last_line = output_text
        else:
            last_line_index = output_text.rfind("\n")
            last_line = output_text[last_line_index:].strip()
        if prompt_type == "score":
            scores = last_line.split(",")
            scores = [int(score.strip().split(":")[-1]) for score in scores]
            if len(scores) != expected_score_num:
                return None
            return scores
        elif prompt_type == "ranking":
            if not _validate_ranking(last_line, expected_score_num):
                return None
            scores_dcit = _ranking_to_scores(last_line)
            scores = [scores_dcit[candidate] for candidate in candidate_identifiers[:len(scores_dcit)]]
            if len(scores) != expected_score_num:
                return None
            return scores
        elif prompt_type == "ranking_score":
            score_dict = parse_score_text(last_line)
            if score_dict is None:
                return None
            scores = [score_dict[candidate] for candidate in candidate_identifiers[:len(score_dict)]]
            if len(scores) != expected_score_num:
                return None
            return scores
        else:
            raise ValueError(f"prompt_type must be one of {Task_format.keys()}")
    except Exception:
        return None


def func_call(
    model_path: str,
    src_list: list[str],
    mt_list: list[list[str]],
    src_langs: Union[str, List[str]],
    trg_langs: Union[str, List[str]],
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 6,
    prompt_type: str = "ranking_score",
    add_example: bool = False,
    model = None,
    tokenizer = None,
    notes_list: Optional[list[str]] = None,
):
    from vllm import LLM, SamplingParams

    assert prompt_type in Task_format.keys()

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)

    if len(src_list) != len(mt_list) or len(src_list) != len(src_langs) or len(src_list) != len(trg_langs):
        raise ValueError("src_list, mt_list, src_langs, and trg_langs must have the same length.")
    if notes_list is not None and len(notes_list) != len(src_list):
        raise ValueError("notes_list must have the same length as src_list.")

    if model is None or tokenizer is None:
        gen_rm, tokenizer = load_model_tokenizer(model_path)
    else:
        gen_rm = model
        tokenizer = tokenizer
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

    # Build prompts
    prompt_list = []
    for i, (src_text, mt_texts, src_lang, trg_lang) in enumerate(zip(src_list, mt_list, src_langs, trg_langs)):
        notes = notes_list[i] if notes_list else None
        prompt = get_GQM_with_notes_prompt(src_lang, trg_lang, src_text, mt_texts, prompt_type, add_example=add_example, notes=notes)
        messages = [
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_list.append(input_text)
    
    # Initial inference
    outputs = gen_rm.generate(prompt_list, sampling_params)
    output_text_list = [output.outputs[0].text for output in outputs]
    score_list = [extract_score(output_text, prompt_type, len(mt_texts)) for output_text, mt_texts in zip(output_text_list, mt_list)]

    # Retry loop
    retry_count = 0
    failed_indices = [i for i, s in enumerate(score_list) if s is None]

    while failed_indices and retry_count < retry:
        retry_count += 1
        print(f"Retry attempt {retry_count}: {len(failed_indices)} failed items remaining...")

        retry_prompts = [prompt_list[i] for i in failed_indices]
        retry_mt_list = [mt_list[i] for i in failed_indices]

        retry_sampling_params = SamplingParams(temperature=1.0, top_p=top_p, max_tokens=max_new_tokens)
        retry_outputs = gen_rm.generate(retry_prompts, retry_sampling_params)
        retry_texts = [output.outputs[0].text for output in retry_outputs]
        retry_scores = [extract_score(text, prompt_type, len(mt_texts)) for text, mt_texts in zip(retry_texts, retry_mt_list)]

        # Replace failed entries
        for idx, new_text, new_score in zip(failed_indices, retry_texts, retry_scores):
            output_text_list[idx] = new_text
            score_list[idx] = new_score

        # Recalculate failed indices
        failed_indices = [i for i, s in enumerate(score_list) if s is None]

    if failed_indices:
        print(f"Warning: {len(failed_indices)} items still failed after {retry} retries.")

    return {"scores": score_list, "responses": output_text_list}
