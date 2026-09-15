import json
import re
from pathlib import Path
from typing import Callable, Optional, Union

from inference.inst_flash_gpe_prompts import (
    build_candidate_prompt,
    build_post_edit_prompt,
    validate_prompt_type,
)
from utils.helpers import get_auto_tp_size


class InstructEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def init_inst_model(model_path: str, **vllm_kwargs) -> InstructEngine:
    from vllm import LLM

    tp_size = get_auto_tp_size()
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        **vllm_kwargs,
    )
    return InstructEngine(model, model.get_tokenizer())


def _render_prompt(engine: InstructEngine, prompt: str, enable_thinking: bool) -> str:
    return engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def split_thinking(text: str) -> tuple[Optional[str], str]:
    text = (text or "").strip()
    match = re.match(r"^\s*<think>(.*?)</think>\s*(.*)$", text, re.DOTALL)
    if match:
        return match.group(1).strip() or None, match.group(2).strip()
    if "</think>" in text:
        thinking, answer = text.split("</think>", 1)
        return thinking.removeprefix("<think>").strip() or None, answer.strip()
    return None, text


def extract_candidate_response(
    response: str, max_candidates: int, *, explicit_analysis: bool = False
) -> Optional[list[str]]:
    if not isinstance(response, str):
        return None
    if explicit_analysis:
        if not response.startswith("# Step-by-step Analysis"):
            return None
        first_candidate = response.find("# Candidate ")
        if first_candidate < 0 or not response[:first_candidate].strip():
            return None
        analysis = response[len("# Step-by-step Analysis"):first_candidate].strip()
        if not analysis:
            return None
        response = response[first_candidate:]
    matches = list(re.finditer(r"(?m)^# Candidate ([1-9][0-9]*)[ \t]*$", response))
    if len(matches) != max_candidates:
        return None
    if response[:matches[0].start()].strip():
        return None
    translations = []
    for index, match in enumerate(matches):
        if int(match.group(1)) != index + 1:
            return None
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        value = response[match.end():end].strip()
        if not value or "```" in value or value.startswith("# "):
            return None
        translations.append(value)
    normalized = {" ".join(item.split()).casefold() for item in translations}
    return translations if len(normalized) == len(translations) else None


def extract_post_edit_response(
    response: str, *, explicit_analysis: bool = False
) -> Optional[str]:
    if not isinstance(response, str):
        return None
    marker = "# Final Translation"
    if explicit_analysis:
        analysis_marker = "# Step-by-step Analysis"
        if not response.startswith(analysis_marker):
            return None
        if marker not in response or response.count(marker) != 1:
            return None
        analysis = response[len(analysis_marker):response.index(marker)].strip()
        if not analysis:
            return None
    elif not response.startswith(marker) or response.count(marker) != 1:
        return None
    value = response.split(marker, 1)[1].strip()
    if not value or value.startswith("# ") or "```" in value:
        return None
    return value


def _generate_with_retries(
    engine: InstructEngine,
    prompts: list[str],
    parser: Callable[[str], object],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    repetition_penalty: float,
    max_tokens: int,
    retry: int,
    enable_thinking: bool,
) -> list[dict]:
    from vllm import SamplingParams

    rendered = [_render_prompt(engine, prompt, enable_thinking) for prompt in prompts]
    results = [{"parsed": None, "response": None, "thinking": None,
                "raw_output": None, "output_tokens": None} for _ in prompts]
    for attempt in range(retry + 1):
        pending = [i for i, result in enumerate(results) if result["parsed"] is None]
        if not pending:
            break
        params = SamplingParams(
            temperature=min(1.0, temperature + 0.1 * attempt),
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )
        outputs = engine.model.generate([rendered[i] for i in pending], params)
        for index, output in zip(pending, outputs):
            raw = output.outputs[0].text
            token_ids = getattr(output.outputs[0], "token_ids", None)
            output_tokens = len(token_ids) if token_ids is not None else None
            thinking, answer = split_thinking(raw)
            try:
                parsed = parser(answer, index)
            except TypeError:
                parsed = parser(answer)
            if parsed is None:
                try:
                    parsed = parser(raw, index)
                except TypeError:
                    parsed = parser(raw)
            results[index] = {
                "parsed": parsed,
                "response": answer,
                "thinking": thinking,
                "raw_output": raw,
                "output_tokens": output_tokens,
            }
    return results


def _normalize_languages(size: int, src_langs: Union[str, list[str]], trg_langs: Union[str, list[str]]):
    src_langs = [src_langs] * size if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * size if isinstance(trg_langs, str) else trg_langs
    if len(src_langs) != size or len(trg_langs) != size:
        raise ValueError("All input lists must have the same length")
    return src_langs, trg_langs


def run_candidate_generation_stage(src_list, src_langs, trg_langs, *, max_candidates=8,
                                   candidate_counts=None,
                                   model, temperature=1.0, top_p=0.95, top_k=20,
                                   presence_penalty=1.5, repetition_penalty=1.0,
                                   max_tokens=8192,
                                   retry=3, enable_thinking=True):
    validate_prompt_type("markdown", max_candidates)
    src_langs, trg_langs = _normalize_languages(len(src_list), src_langs, trg_langs)
    if candidate_counts is None:
        candidate_counts = [max_candidates] * len(src_list)
    if len(candidate_counts) != len(src_list):
        raise ValueError("candidate_counts must match src_list length")
    prompts = [build_candidate_prompt(
                   sl, tl, source, max_candidates, count,
                   explicit_analysis=not enable_thinking)
               for source, sl, tl, count in zip(src_list, src_langs, trg_langs, candidate_counts)]
    results = _generate_with_retries(
        model, prompts,
        lambda text, index: extract_candidate_response(
            text, candidate_counts[index], explicit_analysis=not enable_thinking),
        temperature=temperature, top_p=top_p, top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens, retry=retry,
        enable_thinking=enable_thinking,
    )
    return {
        "prompts": prompts,
        "translations": [r["parsed"] or [] for r in results],
        "responses": [r["response"] for r in results],
        "raw_outputs": [r["raw_output"] for r in results],
        "thinking": [r["thinking"] for r in results],
        "output_tokens": [r["output_tokens"] for r in results],
    }


def run_pipeline(src_list, src_langs, trg_langs, *, model, max_candidates=8,
                 candidate_counts=None,
                 candidate_temperature=1.0, candidate_top_p=0.95,
                 candidate_top_k=20, candidate_presence_penalty=1.5,
                 candidate_repetition_penalty=1.0, candidate_max_tokens=8192,
                 post_edit_temperature=1.0, post_edit_top_p=0.95,
                 post_edit_top_k=20, post_edit_presence_penalty=1.5,
                 post_edit_repetition_penalty=1.0, post_edit_max_tokens=8192,
                 retry=3, enable_thinking=True):
    candidate = run_candidate_generation_stage(
        src_list, src_langs, trg_langs, max_candidates=max_candidates,
        candidate_counts=candidate_counts, model=model,
        temperature=candidate_temperature, top_p=candidate_top_p,
        top_k=candidate_top_k,
        presence_penalty=candidate_presence_penalty,
        repetition_penalty=candidate_repetition_penalty,
        max_tokens=candidate_max_tokens, retry=retry, enable_thinking=enable_thinking,
    )
    valid = [i for i, values in enumerate(candidate["translations"]) if len(values) >= 2]
    post = {"translations": [None] * len(src_list), "responses": [None] * len(src_list),
            "raw_outputs": [None] * len(src_list), "thinking": [None] * len(src_list),
            "output_tokens": [None] * len(src_list), "prompts": [None] * len(src_list)}
    if valid:
        prompts = [build_post_edit_prompt(
            src_langs[i], trg_langs[i], src_list[i], candidate["translations"][i],
            explicit_analysis=not enable_thinking,
        ) for i in valid]
        results = _generate_with_retries(
            model, prompts,
            lambda text, index: extract_post_edit_response(
                text, explicit_analysis=not enable_thinking),
            temperature=post_edit_temperature,
            top_p=post_edit_top_p, top_k=post_edit_top_k,
            presence_penalty=post_edit_presence_penalty,
            repetition_penalty=post_edit_repetition_penalty,
            max_tokens=post_edit_max_tokens, retry=retry,
            enable_thinking=enable_thinking,
        )
        for local, original in enumerate(valid):
            post["prompts"][original] = prompts[local]
            post["translations"][original] = results[local]["parsed"]
            post["responses"][original] = results[local]["response"]
            post["raw_outputs"][original] = results[local]["raw_output"]
            post["thinking"][original] = results[local]["thinking"]
            post["output_tokens"][original] = results[local]["output_tokens"]
    return {"candidate_generation": candidate,
            "usable_candidate_counts": [len(v) for v in candidate["translations"]],
            "post_edit": post}


def main(input_path: str, output_path: str, model_path: str, max_samples: int = 0,
         max_candidates: int = 8, candidate_temperature: float = 1.0,
         candidate_top_p: float = 0.95, candidate_top_k: int = 20,
         candidate_presence_penalty: float = 1.5,
         candidate_repetition_penalty: float = 1.0,
         candidate_max_tokens: int = 8192,
         post_edit_temperature: float = 1.0, post_edit_top_p: float = 0.95,
         post_edit_top_k: int = 20, post_edit_presence_penalty: float = 1.5,
         post_edit_repetition_penalty: float = 1.0,
         post_edit_max_tokens: int = 8192, retry: int = 3, enable_thinking: bool = True,
         gpu_memory_utilization: float = 0.9, max_model_len: int = 32768):
    import pandas as pd
    frame = pd.read_parquet(input_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    model = init_inst_model(model_path, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
    result = run_pipeline(frame.src_text.tolist(), frame.src_lang.tolist(), frame.trg_lang.tolist(), model=model,
                          max_candidates=max_candidates, candidate_temperature=candidate_temperature,
                          candidate_top_p=candidate_top_p, candidate_top_k=candidate_top_k,
                          candidate_presence_penalty=candidate_presence_penalty,
                          candidate_repetition_penalty=candidate_repetition_penalty,
                          candidate_max_tokens=candidate_max_tokens,
                          post_edit_temperature=post_edit_temperature, post_edit_top_p=post_edit_top_p,
                          post_edit_top_k=post_edit_top_k,
                          post_edit_presence_penalty=post_edit_presence_penalty,
                          post_edit_repetition_penalty=post_edit_repetition_penalty,
                          post_edit_max_tokens=post_edit_max_tokens, retry=retry, enable_thinking=enable_thinking)
    items = []
    for i, row in frame.iterrows():
        items.append({"index": i, "src_text": row.src_text, "ref_text": row.get("trg_text"),
                      "src_lang": row.src_lang, "trg_lang": row.trg_lang,
                      "candidates": result["candidate_generation"]["translations"][i],
                      "candidate_prompt": result["candidate_generation"]["prompts"][i],
                      "candidate_response": result["candidate_generation"]["responses"][i],
                      "candidate_thinking": result["candidate_generation"]["thinking"][i],
                      "candidate_output_tokens": result["candidate_generation"]["output_tokens"][i],
                      "usable_candidate_count": result["usable_candidate_counts"][i],
                      "flash_gpe_translation": result["post_edit"]["translations"][i],
                      "flash_gpe_prompt": result["post_edit"]["prompts"][i],
                      "flash_gpe_response": result["post_edit"]["responses"][i],
                      "flash_gpe_thinking": result["post_edit"]["thinking"][i],
                      "flash_gpe_output_tokens": result["post_edit"]["output_tokens"][i]})
    payload = {"method": "inst_flash_gpe", "model_path": model_path,
               "settings": {"prompt_type": "markdown", "max_candidates": max_candidates,
                             "enable_thinking": enable_thinking, "retry": retry,
                             "candidate_temperature": candidate_temperature,
                             "candidate_top_p": candidate_top_p,
                             "candidate_top_k": candidate_top_k,
                             "candidate_presence_penalty": candidate_presence_penalty,
                             "candidate_repetition_penalty": candidate_repetition_penalty,
                             "candidate_max_tokens": candidate_max_tokens,
                             "post_edit_temperature": post_edit_temperature,
                             "post_edit_top_p": post_edit_top_p,
                             "post_edit_top_k": post_edit_top_k,
                             "post_edit_presence_penalty": post_edit_presence_penalty,
                             "post_edit_repetition_penalty": post_edit_repetition_penalty,
                             "post_edit_max_tokens": post_edit_max_tokens}, "items": items}
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(items)} items to {destination}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
