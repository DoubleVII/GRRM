from typing import Union, List, Optional
from utils.config import LANG_MAP
from utils.helpers import get_auto_tp_size


def load_model_tokenizer(model_path: str):
    from vllm import LLM
    print(f"loading model from {model_path}...")
    tp_size = get_auto_tp_size()
    print(f"Detected CUDA devices: {tp_size}; tensor_parallel_size={tp_size}")
    model = LLM(model_path, tensor_parallel_size=tp_size)
    tokenizer = model.get_tokenizer()
    return model, tokenizer




def _block_extractor(response:str) -> str:
    empty_out = None

    response = response.strip()
    if not response:
        return empty_out
    if not response.endswith("```"):
        return empty_out
    response = response[:-3]
    block_start = response.rfind("```")
    if block_start == -1:
        return empty_out
    extract_out = response[block_start+3:].strip()
    if not extract_out:
        return empty_out
    return extract_out

def _ssr_extractor(response: str) -> str:
    """Extract the final answer from SSR-style outputs.

    The SSR models are instructed to wrap the reasoning in <think>...</think>
    and the final answer in <answer>...</answer>. We only return the content
    inside the <answer> tag. Returning None signals a failed extraction and
    will trigger a retry in the caller.
    """
    empty_out = None

    if response is None:
        return empty_out

    response = response.strip()
    if not response:
        return empty_out

    start_tag = "<answer>"
    end_tag = "</answer>"

    start = response.find(start_tag)
    if start == -1:
        return empty_out
    start += len(start_tag)

    end = response.find(end_tag, start)
    if end == -1:
        return empty_out

    extract_out = response[start:end].strip()
    if not extract_out:
        return empty_out
    return extract_out

def _cot_extractor(response: str) -> str:
    """
    seedx output: translation\n[COT] cot text
    """
    cot_tag = "[COT]"
    empty_out = None

    response = response.strip()
    if not response:
        return empty_out
    if cot_tag not in response:
        return response.strip()
    response = response[:response.find(cot_tag)]
    response = response.strip()
    return response

def extract_answer(text: str, prompt_type: str):
    if prompt_type == "Tower.lastl":
        text = text.strip()
        lines = text.split("\n")
        return lines[-1].strip()
    elif prompt_type == "codeblock-think":
        return _block_extractor(text)
    elif prompt_type == "SSR":
        return _ssr_extractor(text)
    elif prompt_type == "seedx-cot":
        return _cot_extractor(text)
    else:
        return text.strip()

def get_prompt(
    prompt_type, src_lang, trg_lang, src_text, trg_token: Optional[str] = None
):
    
    if len(src_lang) == 2:
        src_lang = LANG_MAP[src_lang]
    if len(trg_lang) == 2:
        trg_lang = LANG_MAP[trg_lang]
        trg_token = trg_lang
    if prompt_type == "Tower" or prompt_type == "Tower.lastl":
        src_text = src_text.replace("\n", " ")
        return f"Translate the following text from {src_lang} into {trg_lang}:\n{src_lang}: {src_text}\n{trg_lang}:"
    elif prompt_type == "seedx":
        assert trg_token is not None, "trg_token must be provided for seedx prompt"
        return f"Translate the following {src_lang} sentence into {trg_lang}:\n{src_text} <{trg_token}>"
    elif prompt_type == "seedx-cot":
        assert trg_token is not None, "trg_token must be provided for seedx prompt"
        return f"Translate the following {src_lang} sentence into {trg_lang} and explain it in detail:\n{src_text} <{trg_token}>"
    elif prompt_type == "codeblock-think":
        return f"""Translate the following text from {src_lang} into {trg_lang}. Perform a step by step analysis and output the final translation in a code block.

Source text:
```
{src_text}
```
"""
    elif prompt_type == "SSR":
        # SSR series models expect a specific conversation-style system prompt.
        # We follow the official quickstart format and embed the translation
        # instruction into the "User" message.

        system_prompt = (
            "<|startoftext|>A conversation between User and Assistant. The User asks a question, "
            "and the Assistant solves it. The Assistant first thinks about the reasoning process "
            "in the mind and then provides the User with the answer. The reasoning process is "
            "enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n"
            "User:\n{}\n\nAssistant:\n"
        )

        instruction = f"Translate the following text to {trg_lang}:\n{{}}"

        user_question = instruction.format(src_text)
        return system_prompt.format(user_question)
    else:
        raise NotImplementedError



def func_call(
    model_path: str,
    src_list: list[str],
    src_langs: Union[str, List[str]],
    trg_langs: Union[str, List[str]],
    *,
    sampling_n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 4,
    prompt_type: str = "codeblock-think",
    use_chat_template: bool = True,
    model = None,
    tokenizer = None,
    **kwargs,
):
    from vllm import SamplingParams

    if sampling_n < 1:
        raise ValueError("sampling_n must be at least 1")

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)

    if not (len(src_list) == len(src_langs) == len(trg_langs)):
        raise ValueError("src_list, src_langs, and trg_langs must have the same length.")
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path)


    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=sampling_n,
    )

    # Build prompts
    prompt_list = []
    for src_text, src_lang, trg_lang in zip(src_list, src_langs, trg_langs):
        prompt = get_prompt(prompt_type, src_lang, trg_lang, src_text)
        if use_chat_template:
            messages = [
                {"role": "user", "content": prompt},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_list.append(input_text)
        else:
            prompt_list.append(prompt)
        
    num_inputs = len(prompt_list)

    # Per-input candidate texts and extracted answers (one list per input)
    raw_output_text_nested: list[list[str]] = [[] for _ in range(num_inputs)]
    output_text_nested: list[list[Optional[str]]] = [[] for _ in range(num_inputs)]

    # Unified initial + retry loop.
    # attempt == 0: generate for all inputs.
    # attempt  > 0: re-generate only for inputs where any candidate failed extraction.
    for attempt in range(retry + 1):
        if attempt == 0:
            run_indices = list(range(num_inputs))
        else:
            run_indices = [
                i for i, cand_list in enumerate(output_text_nested)
                if any(ans is None for ans in cand_list)
            ]

        if not run_indices:
            break

        if attempt > 0:
            sampling_params.temperature = min(1.0, sampling_params.temperature + 0.1)

        batch_prompts = [prompt_list[i] for i in run_indices]
        batch_outputs = model.generate(batch_prompts, sampling_params)

        for src_idx, output in zip(run_indices, batch_outputs):
            raw_texts = []
            answers = []
            for candidate in output.outputs:
                text = candidate.text
                raw_texts.append(text)
                answers.append(extract_answer(text, prompt_type))

            raw_output_text_nested[src_idx] = raw_texts
            output_text_nested[src_idx] = answers

    # Placeholder for unresolved None outputs after retries
    output_text_nested = [
        [text if text is not None else "Translation Failed." for text in cand_list]
        for cand_list in output_text_nested
    ]

    # Backward-compatible return shape:
    # - If sampling_n == 1: responses/raw_outputs are flat lists (one per input)
    # - If sampling_n > 1: responses/raw_outputs are nested lists [num_inputs][sampling_n]
    if sampling_n == 1:
        responses = [cand_list[0] for cand_list in output_text_nested]
        raw_outputs = [cand_list[0] for cand_list in raw_output_text_nested]
    else:
        responses = output_text_nested
        raw_outputs = raw_output_text_nested

    return {"responses": responses, "raw_outputs": raw_outputs}
