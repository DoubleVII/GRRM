from typing import Union, List, Optional
from utils.config import LANG_MAP
from utils.helpers import get_auto_tp_size


def load_model_tokenizer(model_path: str, **vllm_kwargs):
    from vllm import LLM
    print(f"loading model from {model_path}...")
    tp_size = get_auto_tp_size()
    print(f"Detected CUDA devices: {tp_size}; tensor_parallel_size={tp_size}")
    model = LLM(model_path, tensor_parallel_size=tp_size, **vllm_kwargs)
    tokenizer = model.get_tokenizer()
    return model, tokenizer

promt_template = """Given a source text in {} and a translation text in {}. Perform a step by step analysis of translation quality and assign a score on a scale from 0 to 10.
Source text:
```
{}
```

Translation text:
```
{}
```
"""


def extract_score(output_text: str) -> Optional[int]:
    output_text = output_text.strip()
    try:
        last_line_index = output_text.rfind("\n")
        last_line = output_text[last_line_index:].strip()
        score = int(last_line)
        return score
    except Exception:
        return None


def func_call(
    model_path: str,
    src_list: list[str],
    mt_list: list[str],
    src_langs: Union[str, List[str]],
    trg_langs: Union[str, List[str]],
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 6,
    model = None,
    tokenizer = None,
):
    from vllm import SamplingParams

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)

    if len(src_list) != len(mt_list) or len(src_list) != len(src_langs) or len(src_list) != len(trg_langs):
        raise ValueError("src_list, mt_list, src_langs, and trg_langs must have the same length.")
    
    if model is None or tokenizer is None:
        gen_rm, tokenizer = load_model_tokenizer(model_path)
    else:
        gen_rm = model
        tokenizer = tokenizer
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

    # Build prompts
    prompt_list = []
    for src_text, mt_text, src_lang, trg_lang in zip(src_list, mt_list, src_langs, trg_langs):
        if len(src_lang) == 2:
            src_lang = LANG_MAP[src_lang]
        if len(trg_lang) == 2:
            trg_lang = LANG_MAP[trg_lang]
        prompt = promt_template.format(src_lang, trg_lang, src_text, mt_text)
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
    score_list = [extract_score(output_text) for output_text in output_text_list]

    # Retry loop
    retry_count = 0
    failed_indices = [i for i, s in enumerate(score_list) if s is None]

    while failed_indices and retry_count < retry:
        retry_count += 1
        print(f"Retry attempt {retry_count}: {len(failed_indices)} failed items remaining...")

        retry_prompts = [prompt_list[i] for i in failed_indices]
        retry_sampling_params = SamplingParams(temperature=1.0, top_p=top_p, max_tokens=max_new_tokens)
        retry_outputs = gen_rm.generate(retry_prompts, retry_sampling_params)
        retry_texts = [output.outputs[0].text for output in retry_outputs]
        retry_scores = [extract_score(text) for text in retry_texts]

        # Replace failed entries
        for idx, new_text, new_score in zip(failed_indices, retry_texts, retry_scores):
            output_text_list[idx] = new_text
            score_list[idx] = new_score

        # Recalculate failed indices
        failed_indices = [i for i, s in enumerate(score_list) if s is None]

    if failed_indices:
        print(f"Warning: {len(failed_indices)} items still failed after {retry} retries.")

    return {"scores": score_list, "responses": output_text_list}
