import json
import re
from pathlib import Path
from typing import Callable, Optional, Union

from inference.inst_mt_prompts import build_translation_prompt
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


def _render_prompt(
    engine: InstructEngine, prompt: str, enable_thinking: bool
) -> str:
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


def extract_translation_response(response: str) -> Optional[dict[str, str]]:
    if not isinstance(response, str):
        return None
    analysis_marker = "# Step-by-step Analysis"
    translation_marker = "# Final Translation"
    if not response.startswith(analysis_marker):
        return None
    if response.count(analysis_marker) != 1 or response.count(translation_marker) != 1:
        return None
    marker_index = response.index(translation_marker)
    analysis = response[len(analysis_marker):marker_index].strip()
    translation = response[marker_index + len(translation_marker):].strip()
    if not analysis or not translation:
        return None
    if "```" in analysis or "```" in translation or translation.startswith("# "):
        return None
    return {"analysis": analysis, "translation": translation}


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

    rendered = [
        _render_prompt(engine, prompt, enable_thinking) for prompt in prompts
    ]
    results = [{
        "parsed": None,
        "response": None,
        "thinking": None,
        "raw_output": None,
        "output_tokens": None,
    } for _ in prompts]
    for attempt in range(retry + 1):
        pending = [
            index for index, result in enumerate(results)
            if result["parsed"] is None
        ]
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
        outputs = engine.model.generate(
            [rendered[index] for index in pending], params
        )
        for index, output in zip(pending, outputs):
            generated = output.outputs[0]
            raw = generated.text
            token_ids = getattr(generated, "token_ids", None)
            thinking, answer = split_thinking(raw)
            parsed = parser(answer)
            if parsed is None:
                parsed = parser(raw)
            results[index] = {
                "parsed": parsed,
                "response": answer,
                "thinking": thinking,
                "raw_output": raw,
                "output_tokens": (
                    len(token_ids) if token_ids is not None else None
                ),
            }
    return results


def _normalize_languages(
    size: int,
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
) -> tuple[list[str], list[str]]:
    src_langs = [src_langs] * size if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * size if isinstance(trg_langs, str) else trg_langs
    if len(src_langs) != size or len(trg_langs) != size:
        raise ValueError("All input lists must have the same length")
    return src_langs, trg_langs


def run_translation_stage(
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    model: InstructEngine,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
    max_tokens: int = 8192,
    retry: int = 3,
    enable_thinking: bool = False,
) -> dict:
    src_langs, trg_langs = _normalize_languages(
        len(src_list), src_langs, trg_langs
    )
    prompts = [
        build_translation_prompt(src_lang, trg_lang, source)
        for source, src_lang, trg_lang in zip(src_list, src_langs, trg_langs)
    ]
    results = _generate_with_retries(
        model,
        prompts,
        extract_translation_response,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        retry=retry,
        enable_thinking=enable_thinking,
    )
    return {
        "prompts": prompts,
        "translations": [
            result["parsed"]["translation"] if result["parsed"] else None
            for result in results
        ],
        "analyses": [
            result["parsed"]["analysis"] if result["parsed"] else None
            for result in results
        ],
        "responses": [result["response"] for result in results],
        "raw_outputs": [result["raw_output"] for result in results],
        "thinking": [result["thinking"] for result in results],
        "output_tokens": [result["output_tokens"] for result in results],
    }


def main(
    input_path: str,
    output_path: str,
    model_path: str,
    max_samples: int = 0,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
    max_tokens: int = 8192,
    retry: int = 3,
    enable_thinking: bool = False,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    import pandas as pd

    frame = pd.read_parquet(input_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    model = init_inst_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    result = run_translation_stage(
        frame["src_text"].tolist(),
        frame["src_lang"].tolist(),
        frame["trg_lang"].tolist(),
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        retry=retry,
        enable_thinking=enable_thinking,
    )
    items = []
    for index, row in frame.iterrows():
        items.append({
            "index": index,
            "src_text": row["src_text"],
            "ref_text": row.get("trg_text"),
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "prompt": result["prompts"][index],
            "analysis": result["analyses"][index],
            "translation": result["translations"][index],
            "response": result["responses"][index],
            "thinking": result["thinking"][index],
            "output_tokens": result["output_tokens"][index],
        })
    payload = {
        "method": "inst_mt",
        "model_path": model_path,
        "settings": {
            "enable_thinking": enable_thinking,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "retry": retry,
        },
        "items": items,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(items)} items to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
