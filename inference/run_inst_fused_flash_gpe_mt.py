"""Single-generation instruct FlashGPE with explicit candidate comparison."""

import json
import re
from pathlib import Path
from typing import Optional

from inference.inst_flash_gpe_prompts import build_fused_prompt, validate_prompt_type
from inference.run_inst_flash_gpe_mt import (
    _generate_with_retries,
    _normalize_languages,
    extract_candidate_response,
    extract_post_edit_response,
    init_inst_model,
)


def extract_fused_response(response: str, candidate_count: int) -> Optional[dict]:
    if not isinstance(response, str):
        return None
    analysis_heading = "# Step-by-step Analysis"
    comparison_heading = "# Candidate Comparison"
    final_heading = "# Final Translation"
    if not response.startswith(analysis_heading):
        return None
    if response.count(comparison_heading) != 1 or response.count(final_heading) != 1:
        return None
    first_candidate = response.find("# Candidate 1")
    comparison = response.find(comparison_heading)
    final = response.find(final_heading)
    if not (len(analysis_heading) < first_candidate < comparison < final):
        return None
    analysis = response[len(analysis_heading):first_candidate].strip()
    comparison_text = response[comparison + len(comparison_heading):final].strip()
    if not analysis or not comparison_text or comparison_text.startswith("# "):
        return None
    other_sections = (
        analysis + "\n" + comparison_text + "\n"
        + response[final + len(final_heading):]
    )
    if re.search(r"(?m)^#[ \t]", other_sections):
        return None
    candidate_section = response[first_candidate:comparison]
    headings = re.findall(r"(?m)^#[ \t]+(.+?)[ \t]*$", candidate_section)
    if headings != [f"Candidate {i}" for i in range(1, candidate_count + 1)]:
        return None
    candidates = extract_candidate_response(
        candidate_section, candidate_count
    )
    final_text = response[final + len(final_heading):].strip()
    if not final_text or final_text.startswith("# ") or "```" in final_text:
        return None
    translation = final_text
    if candidates is None:
        return None
    return {
        "analysis": analysis,
        "candidates": candidates,
        "comparison": comparison_text,
        "translation": translation,
    }


def run_pipeline(
    src_list, src_langs, trg_langs, *, model, candidate_count=4,
    temperature=1.0, top_p=1.0, top_k=0, presence_penalty=0.0,
    repetition_penalty=1.0, max_tokens=8192, retry=3, enable_thinking=True,
):
    validate_prompt_type("markdown", candidate_count)
    src_langs, trg_langs = _normalize_languages(len(src_list), src_langs, trg_langs)
    prompts = [
        build_fused_prompt(sl, tl, source, candidate_count)
        for source, sl, tl in zip(src_list, src_langs, trg_langs)
    ]
    results = _generate_with_retries(
        model, prompts, lambda text: extract_fused_response(text, candidate_count),
        temperature=temperature, top_p=top_p, top_k=top_k,
        presence_penalty=presence_penalty, repetition_penalty=repetition_penalty,
        max_tokens=max_tokens, retry=retry, enable_thinking=enable_thinking,
    )
    return {
        "prompts": prompts,
        "translations": [r["parsed"]["translation"] if r["parsed"] else None for r in results],
        "candidates": [r["parsed"]["candidates"] if r["parsed"] else [] for r in results],
        "analyses": [r["parsed"]["analysis"] if r["parsed"] else None for r in results],
        "comparisons": [r["parsed"]["comparison"] if r["parsed"] else None for r in results],
        "responses": [r["response"] for r in results],
        "raw_outputs": [r["raw_output"] for r in results],
        "thinking": [r["thinking"] for r in results],
        "output_tokens": [r["output_tokens"] for r in results],
    }


def main(
    input_path: str, output_path: str, model_path: str, max_samples: int = 0,
    candidate_count: int = 4, temperature: float = 1.0, top_p: float = 1.0,
    top_k: int = 0, presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0, max_tokens: int = 8192,
    retry: int = 3, enable_thinking: bool = True,
    gpu_memory_utilization: float = 0.9, max_model_len: int = 32768,
):
    import pandas as pd

    frame = pd.read_parquet(input_path)
    missing = sorted({"src_text", "src_lang", "trg_lang"} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    engine = init_inst_model(
        model_path, gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    result = run_pipeline(
        frame.src_text.tolist(), frame.src_lang.tolist(), frame.trg_lang.tolist(),
        model=engine, candidate_count=candidate_count, temperature=temperature,
        top_p=top_p, top_k=top_k, presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty, max_tokens=max_tokens,
        retry=retry, enable_thinking=enable_thinking,
    )
    items = [
        {
            "index": i, "src_text": row.src_text, "ref_text": row.get("trg_text"),
            "src_lang": row.src_lang, "trg_lang": row.trg_lang,
            "candidates": result["candidates"][i], "analysis": result["analyses"][i],
            "comparison": result["comparisons"][i],
            "fused_flash_gpe_translation": result["translations"][i],
            "fused_flash_gpe_prompt": result["prompts"][i],
            "fused_flash_gpe_response": result["responses"][i],
            "fused_flash_gpe_raw_output": result["raw_outputs"][i],
            "fused_flash_gpe_thinking": result["thinking"][i],
            "fused_flash_gpe_output_tokens": result["output_tokens"][i],
        }
        for i, row in frame.iterrows()
    ]
    payload = {
        "method": "inst_fused_flash_gpe", "model_path": model_path,
        "settings": {
            "candidate_count": candidate_count, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty, "max_tokens": max_tokens,
            "retry": retry, "enable_thinking": enable_thinking,
        },
        "parse_success_count": sum(value is not None for value in result["translations"]),
        "items": items,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(items)} items to {destination} (parsed={payload['parse_success_count']})")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
