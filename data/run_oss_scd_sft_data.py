from pathlib import Path

import fire
import pandas as pd

from data.oss_mt_sft_data_utils import serialize_parsed, validate_scd_record
from inference.oss_diverse_mt_prompts import (
    build_convergent_prompt,
    build_divergent_prompt,
)
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_diverse_mt import normalize_bool, run_pipeline


METHOD_NAME = "Segment-Level Candidate Deliberation"
METHOD_ID = "scd"


def main(
    data_path: str,
    output_path: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
    reasoning_effort: str = "medium",
    prompt_type: str = "json",
    polish: bool = True,
    min_candidates: int = 3,
    max_candidates: int = 6,
    candidate_confidence: bool = False,
    stage1_temperature: float = 0.8,
    stage1_top_p: float = 0.95,
    stage1_max_tokens: int = 8192,
    stage2_temperature: float = 0.3,
    stage2_top_p: float = 0.8,
    stage2_max_tokens: int = 4096,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Collect parser-validated two-stage SCD supervision from gpt-oss."""
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    if prompt_type not in {"json", "codeblock"}:
        raise ValueError("prompt_type must be json or codeblock")
    polish = normalize_bool(polish, "polish")
    candidate_confidence = normalize_bool(
        candidate_confidence, "candidate_confidence"
    )

    frame = pd.read_parquet(data_path)
    required = {src_key, src_lang_key, trg_lang_key}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)

    sources = frame[src_key].tolist()
    src_langs = frame[src_lang_key].tolist()
    trg_langs = frame[trg_lang_key].tolist()
    model = init_oss_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    output = run_pipeline(
        sources,
        src_langs,
        trg_langs,
        model=model,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        prompt_type=prompt_type,
        polish=polish,
        candidate_confidence=candidate_confidence,
        divergent_temperature=stage1_temperature,
        divergent_top_p=stage1_top_p,
        final_temperature=stage2_temperature,
        final_top_p=stage2_top_p,
        stage1_max_tokens=stage1_max_tokens,
        final_max_tokens=stage2_max_tokens,
        retry=retry,
    )

    records = []
    invalid_count = 0
    for index, row in frame.iterrows():
        analysis = output["divergent"]["analyses"][index]
        stage1_response = output["divergent"]["responses"][index]
        stage1_thinking = output["divergent"]["thinking"][index]
        stage2_response = output["convergent"]["responses"][index]
        stage2_thinking = output["convergent"]["thinking"][index]
        translation = output["convergent"]["translations"][index]
        valid = validate_scd_record(
            stage1_response=stage1_response,
            stage1_thinking=stage1_thinking,
            stage1_analysis=analysis,
            stage2_response=stage2_response,
            stage2_thinking=stage2_thinking,
            stage2_translation=translation,
            prompt_type=prompt_type,
            candidate_confidence=candidate_confidence,
        )
        if not valid:
            invalid_count += 1
            continue
        stage1_prompt = build_divergent_prompt(
            row[src_lang_key],
            row[trg_lang_key],
            row[src_key],
            min_candidates,
            max_candidates,
            prompt_type=prompt_type,
            candidate_confidence=candidate_confidence,
        )
        stage2_prompt = build_convergent_prompt(
            row[src_lang_key],
            row[trg_lang_key],
            row[src_key],
            analysis,
            polish=polish,
            candidate_confidence=candidate_confidence,
            prompt_type=prompt_type,
        )
        record = row.to_dict()
        record.update({
            "sft_method": METHOD_ID,
            "sft_method_name": METHOD_NAME,
            "scd_stage1_prompt": stage1_prompt,
            "scd_stage1_thinking": stage1_thinking,
            "scd_stage1_response": stage1_response,
            "scd_stage1_parsed": serialize_parsed(analysis),
            "scd_stage2_prompt": stage2_prompt,
            "scd_stage2_thinking": stage2_thinking,
            "scd_stage2_response": stage2_response,
            "scd_translation": translation,
            "scd_parser_valid": True,
        })
        records.append(record)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(destination, index=False)
    print(
        f"Saved {len(records)} valid {METHOD_ID} rows to {destination}; "
        f"dropped {invalid_count} parser-invalid rows"
    )


if __name__ == "__main__":
    fire.Fire(main)
