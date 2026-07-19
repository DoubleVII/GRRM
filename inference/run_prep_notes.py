from typing import Optional, Union
import warnings

from inference.prompts import get_prep_notes_prompt
from inference.run_rm_SQM import load_model_tokenizer


def _block_extractor(response: str) -> dict:
    empty_out = {"analysis": None, "notes": None}

    response = response.strip()
    if not response or not response.endswith("```"):
        return empty_out
    response = response[:-3]
    block_start = response.rfind("```markdown")
    if block_start == -1:
        return empty_out
    notes = response[block_start + len("```markdown"):].strip()
    analysis = response[:block_start].strip()
    if not notes:
        return empty_out
    return {"analysis": analysis, "notes": notes}


def _difficulty_extractor(response: str) -> int:
    if not response:
        raise ValueError("Difficulty score not found in response")
    for line in reversed(response.strip().splitlines()):
        if line.strip().startswith("Difficulty score:"):
            score_text = line.rsplit(":", 1)[-1].strip()
            if not score_text.endswith("/10"):
                raise ValueError(f"Difficulty score format error: {score_text}")
            try:
                score = int(score_text.split("/", 1)[0])
            except ValueError as exc:
                raise ValueError(f"Difficulty score format error: {score_text}") from exc
            if not 0 <= score <= 10:
                raise ValueError(f"Difficulty score format error: {score_text}")
            return score
    raise ValueError("Difficulty score not found in response")


def no_notes_detect(notes: str) -> bool:
    return "no special translation notes needed" in notes.lower()


def extract_response(response: str) -> Optional[dict]:
    if response is None:
        return None
    response = response.strip()
    extracted = _block_extractor(response)
    analysis = extracted["analysis"]
    notes = extracted["notes"]
    if not analysis or not notes:
        return None
    try:
        difficulty = _difficulty_extractor(analysis)
    except ValueError:
        return None
    return {
        "analysis": analysis,
        "notes": notes,
        "notes_valid": not no_notes_detect(notes),
        "difficulty": difficulty,
    }


def prepare_vllm_inputs(
    src_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    tokenizer,
    use_simple_prompt: bool = True,
) -> list[str]:
    inputs = []
    for src_text, src_lang, trg_lang in zip(src_list, src_langs, trg_langs):
        prompt = get_prep_notes_prompt(
            src_lang,
            trg_lang,
            src_text,
            use_simple_prompt=use_simple_prompt,
        )
        messages = [{"role": "user", "content": prompt}]
        inputs.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return inputs


def run_generate(llm, inputs: list[str], sampling_params) -> list[Optional[dict]]:
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    results = []
    for output in outputs:
        response = output.outputs[0].text
        extracted = extract_response(response)
        if extracted is None:
            results.append(None)
            continue
        results.append({**extracted, "response": response})
    return results


def func_call(
    src_list: list,
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    retry: int = 6,
    model=None,
    tokenizer=None,
    model_path: str = "Qwen/Qwen3-8B",
    use_simple_prompt: bool = True,
):
    from vllm import SamplingParams

    n = len(src_list)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * n
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * n
    if not (len(src_list) == len(src_langs) == len(trg_langs)):
        raise ValueError("src_list, src_langs, and trg_langs must have the same length.")

    if model is None or tokenizer is None:
        model, tokenizer = load_model_tokenizer(model_path)

    inputs = prepare_vllm_inputs(
        src_list,
        src_langs,
        trg_langs,
        tokenizer,
        use_simple_prompt=use_simple_prompt,
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    results = run_generate(model, inputs, sampling_params)
    failed_indices = [i for i, result in enumerate(results) if result is None]
    retry_count = 0
    while failed_indices and retry_count < retry:
        retry_count += 1
        retry_temperature = min(1.0, temperature + 0.2 * retry_count)
        retry_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=retry_temperature,
            top_p=top_p,
        )
        retry_inputs = [inputs[i] for i in failed_indices]
        retry_results = run_generate(model, retry_inputs, retry_params)
        for index, result in zip(failed_indices, retry_results):
            if result is not None:
                results[index] = result
        failed_indices = [i for i, result in enumerate(results) if result is None]

    out_data = {
        "analysis": [],
        "notes": [],
        "response": [],
        "notes_valid": [],
        "difficulty": [],
    }
    for index, result in enumerate(results):
        if result is None:
            warnings.warn(f"Evaluation failed, src_text: {src_list[index]}")
            result = {
                "analysis": None,
                "notes": None,
                "response": None,
                "notes_valid": None,
                "difficulty": None,
            }
        for key in out_data:
            out_data[key].append(result[key])
    return out_data


if __name__ == "__main__":
    pass
