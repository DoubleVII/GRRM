from typing import Union, List, Optional
from utils.config import LANG_MAP
from utils.helpers import extract_before_cot as _cot_extractor


def load_model_tokenizer(model_path: str):
    from vllm import LLM
    print(f"loading model from {model_path}...")
    model = LLM(model_path)
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def extract_answer(text: str, prompt_type: str):
    if prompt_type == "CoT":
        return _cot_extractor(text)
    else:
        return text.strip()

def get_prompt(
    prompt_type, src_lang, trg_lang, src_text, trg_token: Optional[str] = None
):
    assert len(trg_lang) == 2
    trg_token = trg_lang
    if len(src_lang) == 2:
        src_lang = LANG_MAP[src_lang]
    if len(trg_lang) == 2:
        trg_lang = LANG_MAP[trg_lang]
    if prompt_type.lower() == "cot":
        return f"Translate the following {src_lang} sentence into {trg_lang} and explain it in detail:\n{src_text} <{trg_token}>"
    else:
        return f"Translate the following {src_lang} sentence into {trg_lang}:\n{src_text} <{trg_token}>"


def func_call(
    model_path: str,
    src_list: list[str],
    src_langs: Union[str, List[str]],
    trg_langs: Union[str, List[str]],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 4,
    prompt_type: str = "CoT",
    model = None,
    tokenizer = None,
    **kwargs,
):
    from vllm import SamplingParams

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)

    if not (len(src_list) == len(src_langs) == len(trg_langs)):
        raise ValueError("src_list, src_langs, and trg_langs must have the same length.")
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path)

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

    # Build prompts
    prompt_list = []
    for src_text, src_lang, trg_lang in zip(src_list, src_langs, trg_langs):
        prompt = get_prompt(prompt_type, src_lang, trg_lang, src_text)
        prompt_list.append(prompt)
        
    # Initial inference
    outputs = model.generate(prompt_list, sampling_params)
    output_text_list = [output.outputs[0].text for output in outputs]
    # Extract answers
    output_text_list = [extract_answer(text, prompt_type) for text in output_text_list]

    # Retry mechanism: re-generate for None outputs, increase temperature by +0.1 each retry up to 1.0
    while retry:
        to_retry_indices = [i for i, out in enumerate(output_text_list) if out is None]
        if not to_retry_indices:
            break
        sampling_params.temperature = min(1.0, sampling_params.temperature + 0.1)
        retry_prompts = [prompt_list[i] for i in to_retry_indices]
        retry_outputs = model.generate(retry_prompts, sampling_params)
        retry_raw_texts = [output.outputs[0].text for output in retry_outputs]
        retry_answers = [extract_answer(text, prompt_type) for text in retry_raw_texts]
        for idx, ans in zip(to_retry_indices, retry_answers):
            output_text_list[idx] = ans
        retry -= 1

    # Placeholder for unresolved None outputs after retries
    output_text_list = [text if text is not None else "Translation Failed." for text in output_text_list]

    return {"responses": output_text_list}


if __name__ == "__main__":
    print(func_call(
        model_path="Seed-X-PPO-7B",
        src_list=["May the force be with you"],
        src_langs="en",
        trg_langs="zh",
    ))