from typing import Union, List, Optional

from inference.prompts import get_group_post_edit_prompt
from inference.run_mt import _block_extractor
from inference.run_rm_SQM import load_model_tokenizer


def func_call(
    model_path: str,
    src_list: list[str],
    mt_list: list[list[str]],
    src_langs: Union[str, List[str]],
    trg_langs: Union[str, List[str]],
    notes_list: Optional[list[Optional[str]]] = None,
    temperature: float = 0.4,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 6,
    model=None,
    tokenizer=None,
):
    from vllm import SamplingParams

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)
    if notes_list is None:
        notes_list = [None] * len(src_list)

    if not (len(src_list) == len(mt_list) == len(src_langs) == len(trg_langs) == len(notes_list)):
        raise ValueError("src_list, mt_list, src_langs, trg_langs, and notes_list must have the same length.")

    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path)

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

    prompt_list = []
    for src_text, mt_texts, notes, src_lang, trg_lang in zip(src_list, mt_list, notes_list, src_langs, trg_langs):
        prompt = get_group_post_edit_prompt(src_lang, trg_lang, src_text, mt_texts, notes=notes)
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_list.append(input_text)

    outputs = model.generate(prompt_list, sampling_params)
    output_text_list = [output.outputs[0].text for output in outputs]
    post_edit_list = [_block_extractor(text) for text in output_text_list]

    retry_count = 0
    failed_indices = [i for i, pe in enumerate(post_edit_list) if pe is None]

    while failed_indices and retry_count < retry:
        retry_count += 1
        print(f"Retry attempt {retry_count}: {len(failed_indices)} failed items remaining...")

        retry_prompts = [prompt_list[i] for i in failed_indices]
        retry_sampling_params = SamplingParams(temperature=1.0, top_p=top_p, max_tokens=max_new_tokens)
        retry_outputs = model.generate(retry_prompts, retry_sampling_params)
        retry_texts = [output.outputs[0].text for output in retry_outputs]
        retry_post_edits = [_block_extractor(text) for text in retry_texts]

        for idx, new_text, new_pe in zip(failed_indices, retry_texts, retry_post_edits):
            output_text_list[idx] = new_text
            post_edit_list[idx] = new_pe

        failed_indices = [i for i, pe in enumerate(post_edit_list) if pe is None]

    if failed_indices:
        print(f"Warning: {len(failed_indices)} items still failed after {retry} retries.")

    # Placeholder for unresolved None outputs after retries
    post_edit_list = [text if text is not None else "Translation Failed." for text in post_edit_list]

    return {"post_edit_mt": post_edit_list, "responses": output_text_list}
