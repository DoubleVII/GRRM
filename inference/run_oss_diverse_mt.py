import json
import warnings
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Optional, Union

from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyError,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
)

from inference.oss_diverse_mt_prompts import (
    build_convergent_prompt,
    build_divergent_prompt,
)
from inference.run_oss_SQM import init_oss_model, load_encoding


def _system_content(reasoning_effort: Optional[str]) -> SystemContent:
    content = SystemContent.new()
    if reasoning_effort is None:
        return content
    efforts = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
    }
    try:
        content.reasoning_effort = efforts[reasoning_effort.lower()]
    except KeyError as error:
        raise ValueError("reasoning_effort must be one of: low, medium, high") from error
    return content


def extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()
    start = text.find("{")
    if start == -1:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except (JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def extract_codeblock_response(text: str) -> Optional[str]:
    """Validate a prep-notes-style response while preserving its free form."""
    text = (text or "").strip()
    if not text or not text.endswith("```"):
        return None
    closing_start = len(text) - 3
    block_start = text.rfind("```", 0, closing_start)
    if block_start == -1:
        return None
    first_newline = text.find("\n", block_start + 3, closing_start)
    if first_newline == -1:
        return None
    block_content = text[first_newline + 1 : closing_start].strip()
    if not block_content:
        return None
    return text


def validate_divergent_result(value: Optional[dict]) -> Optional[dict]:
    if not isinstance(value, dict) or not isinstance(value.get("source_analysis"), str):
        return None
    segments = value.get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    for expected_id, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            return None
        if segment.get("segment_id") != expected_id:
            return None
        if not isinstance(segment.get("source_span"), str) or not segment["source_span"].strip():
            return None
        if not isinstance(segment.get("analysis"), str):
            return None
        candidates = segment.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                return None
            if not isinstance(candidate.get("translation"), str) or not candidate["translation"].strip():
                return None
            if not isinstance(candidate.get("angle"), str):
                return None
    return value


def extract_final_translation(text: str) -> Optional[str]:
    text = (text or "").strip()
    start_tag = "<final_translation>"
    end_tag = "</final_translation>"
    start = text.find(start_tag)
    if start == -1:
        # Harmony already separates private reasoning from the final channel. Some
        # gpt-oss generations ignore the requested wrapper but otherwise return a
        # clean translation, so the final-channel text itself is a safe fallback.
        if text.startswith("```") and text.endswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 : -3].strip()
        return text or None
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return None
    translation = text[start:end].strip()
    return translation or None


def _prepare_inputs(
    prompts: list[str],
    encoding: HarmonyEncoding,
    reasoning_effort: Optional[str],
) -> list[dict]:
    system_content = _system_content(reasoning_effort)
    inputs = []
    for prompt in prompts:
        conversation = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.USER, prompt),
        ])
        token_ids = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        inputs.append({"prompt_token_ids": token_ids})
    return inputs


def _response_text(token_ids: list[int], encoding: HarmonyEncoding) -> tuple[Optional[str], Optional[str]]:
    try:
        entries = encoding.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)
    except HarmonyError:
        return None, None
    thinking = None
    response = None
    for entry in entries:
        data = entry.to_dict()
        content = data.get("content")
        if not (isinstance(content, list) and content and isinstance(content[0], dict)):
            continue
        text = content[0].get("text")
        channel = data.get("channel")
        if channel == "analysis":
            thinking = text
        elif channel == "final":
            response = text
    # Current Harmony output normally has analysis then final. Keep a fallback for
    # library versions whose serialized dictionaries omit the channel field.
    if response is None and entries:
        data = entries[-1].to_dict()
        content = data.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            response = content[0].get("text")
    return thinking, response


def _generate_with_retries(
    llm,
    inputs: list[dict],
    parser: Callable[[str], Any],
    encoding: HarmonyEncoding,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    retry: int,
) -> list[dict]:
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=encoding.stop_tokens_for_assistant_actions(),
    )
    results: list[Optional[dict]] = [None] * len(inputs)
    last_attempts = [
        {"parsed": None, "response": None, "thinking": None}
        for _ in inputs
    ]
    for attempt in range(retry + 1):
        remaining = [index for index, result in enumerate(results) if result is None]
        if not remaining:
            break
        if attempt:
            params.temperature = min(1.0, params.temperature + 0.1)
        outputs = llm.generate([inputs[index] for index in remaining], sampling_params=params)
        for index, output in zip(remaining, outputs):
            thinking, response = _response_text(
                output.outputs[0].token_ids, encoding
            )
            parsed = parser(response) if response is not None else None
            last_attempts[index] = {
                "parsed": parsed,
                "response": response,
                "thinking": thinking,
            }
            if parsed is not None:
                results[index] = last_attempts[index]
    finalized = []
    for index, result in enumerate(results):
        if result is None:
            warnings.warn(f"OSS generation failed after {retry + 1} attempts for item {index}")
            result = last_attempts[index]
        finalized.append(result)
    return finalized


def run_divergent_stage(
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    model=None,
    model_path: str = "openai/gpt-oss-120b",
    reasoning_effort: Optional[str] = "medium",
    min_candidates: int = 3,
    max_candidates: int = 6,
    prompt_type: str = "json",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    retry: int = 3,
) -> dict:
    n = len(src_list)
    src_langs = [src_langs] * n if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * n if isinstance(trg_langs, str) else trg_langs
    if not (n == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")
    if not (1 <= min_candidates <= max_candidates):
        raise ValueError("Expected 1 <= min_candidates <= max_candidates")
    if prompt_type not in {"json", "codeblock"}:
        raise ValueError("prompt_type must be one of: json, codeblock")
    llm = init_oss_model(model_path) if model is None else model
    encoding = load_encoding()
    prompts = [
        build_divergent_prompt(
            sl,
            tl,
            source,
            min_candidates,
            max_candidates,
            prompt_type=prompt_type,
        )
        for source, sl, tl in zip(src_list, src_langs, trg_langs)
    ]
    parser = (
        (lambda text: validate_divergent_result(extract_json_object(text)))
        if prompt_type == "json"
        else extract_codeblock_response
    )
    results = _generate_with_retries(
        llm,
        _prepare_inputs(prompts, encoding, reasoning_effort),
        parser,
        encoding,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        retry=retry,
    )
    return {
        "analyses": [result["parsed"] for result in results],
        "responses": [result["response"] for result in results],
        "thinking": [result["thinking"] for result in results],
    }


def run_convergent_stage(
    src_list: list[str],
    divergent_results: list,
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    model=None,
    model_path: str = "openai/gpt-oss-120b",
    reasoning_effort: Optional[str] = "medium",
    temperature: float = 0.3,
    top_p: float = 0.8,
    max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    n = len(src_list)
    src_langs = [src_langs] * n if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * n if isinstance(trg_langs, str) else trg_langs
    if not (n == len(divergent_results) == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")
    if any(result is None for result in divergent_results):
        raise ValueError("Convergent stage requires a valid divergent result for every item")
    llm = init_oss_model(model_path) if model is None else model
    encoding = load_encoding()
    prompts = [
        build_convergent_prompt(sl, tl, source, divergent)
        for source, divergent, sl, tl in zip(
            src_list, divergent_results, src_langs, trg_langs
        )
    ]
    results = _generate_with_retries(
        llm,
        _prepare_inputs(prompts, encoding, reasoning_effort),
        extract_final_translation,
        encoding,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        retry=retry,
    )
    return {
        "translations": [result["parsed"] for result in results],
        "responses": [result["response"] for result in results],
        "thinking": [result["thinking"] for result in results],
    }


def run_pipeline(
    src_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    *,
    model,
    model_path: str,
    reasoning_effort: str = "medium",
    min_candidates: int = 3,
    max_candidates: int = 6,
    prompt_type: str = "json",
    divergent_temperature: float = 0.8,
    divergent_top_p: float = 0.95,
    final_temperature: float = 0.3,
    final_top_p: float = 0.8,
    stage1_max_tokens: int = 8192,
    final_max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    divergent = run_divergent_stage(
        src_list,
        src_langs,
        trg_langs,
        model=model,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        prompt_type=prompt_type,
        temperature=divergent_temperature,
        top_p=divergent_top_p,
        max_tokens=stage1_max_tokens,
        retry=retry,
    )

    valid_indices = [
        index for index, analysis in enumerate(divergent["analyses"])
        if analysis is not None
    ]
    convergent_full = {
        "translations": [None] * len(src_list),
        "responses": [None] * len(src_list),
        "thinking": [None] * len(src_list),
    }
    if valid_indices:
        convergent = run_convergent_stage(
            [src_list[index] for index in valid_indices],
            [divergent["analyses"][index] for index in valid_indices],
            [src_langs[index] for index in valid_indices],
            [trg_langs[index] for index in valid_indices],
            model=model,
            model_path=model_path,
            reasoning_effort=reasoning_effort,
            temperature=final_temperature,
            top_p=final_top_p,
            max_tokens=final_max_tokens,
            retry=retry,
        )
        for local_index, original_index in enumerate(valid_indices):
            for key in convergent_full:
                convergent_full[key][original_index] = convergent[key][local_index]

    return {
        "divergent": divergent,
        "convergent": convergent_full,
    }


def main(
    input_path: str,
    output_path: str,
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
    reasoning_effort: str = "medium",
    min_candidates: int = 3,
    max_candidates: int = 6,
    prompt_type: str = "json",
    divergent_temperature: float = 0.8,
    divergent_top_p: float = 0.95,
    final_temperature: float = 0.3,
    final_top_p: float = 0.8,
    stage1_max_tokens: int = 8192,
    final_max_tokens: int = 4096,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Run divergent and convergent translation stages for a parquet file."""
    import pandas as pd

    if prompt_type not in {"json", "codeblock"}:
        raise ValueError("prompt_type must be one of: json, codeblock")
    frame = pd.read_parquet(input_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)

    model = init_oss_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    outputs = run_pipeline(
        frame["src_text"].tolist(),
        frame["src_lang"].tolist(),
        frame["trg_lang"].tolist(),
        model=model,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        prompt_type=prompt_type,
        divergent_temperature=divergent_temperature,
        divergent_top_p=divergent_top_p,
        final_temperature=final_temperature,
        final_top_p=final_top_p,
        stage1_max_tokens=stage1_max_tokens,
        final_max_tokens=final_max_tokens,
        retry=retry,
    )
    items = []
    for index, row in frame.reset_index(drop=True).iterrows():
        items.append({
            "index": index,
            "src_text": row["src_text"],
            "ref_text": row.get("trg_text"),
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "divergent_analysis": outputs["divergent"]["analyses"][index],
            "divergent_response": outputs["divergent"]["responses"][index],
            "two_stage_translation": outputs["convergent"]["translations"][index],
            "two_stage_response": outputs["convergent"]["responses"][index],
        })
    payload = {
        "model_path": model_path,
        "input_path": input_path,
        "settings": {
            "reasoning_effort": reasoning_effort,
            "min_candidates": min_candidates,
            "max_candidates": max_candidates,
            "prompt_type": prompt_type,
            "divergent_temperature": divergent_temperature,
            "divergent_top_p": divergent_top_p,
            "final_temperature": final_temperature,
            "final_top_p": final_top_p,
            "stage1_max_tokens": stage1_max_tokens,
            "final_max_tokens": final_max_tokens,
            "retry": retry,
        },
        "items": items,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
