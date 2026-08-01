from pathlib import Path

import fire
import pandas as pd

from data.flash_gpe_sft_data_utils import validate_flash_gpe_record
from inference.oss_flash_gpe_prompts import (
    build_post_edit_prompt,
    validate_prompt_type,
)
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_flash_gpe_mt import run_pipeline


METHOD_NAME = "FlashGPE"
METHOD_ID = "flash_gpe"


def main(
    data_path: str,
    output_path: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
    max_candidates: int = 4,
    prompt_type: str = "fixed_4",
    reasoning_effort: str = "medium",
    candidate_temperature: float = 0.8,
    candidate_top_p: float = 0.95,
    candidate_max_tokens: int = 4096,
    post_edit_temperature: float = 0.3,
    post_edit_top_p: float = 0.8,
    post_edit_max_tokens: int = 4096,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Collect parser-validated FlashGPE supervision."""
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    validate_prompt_type(prompt_type, max_candidates)
    frame = pd.read_parquet(data_path)
    required = {src_key, src_lang_key, trg_lang_key}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    model = init_oss_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    output = run_pipeline(
        frame[src_key].tolist(),
        frame[src_lang_key].tolist(),
        frame[trg_lang_key].tolist(),
        model=model,
        model_path=model_path,
        max_candidates=max_candidates,
        prompt_type=prompt_type,
        reasoning_effort=reasoning_effort,
        candidate_temperature=candidate_temperature,
        candidate_top_p=candidate_top_p,
        candidate_max_tokens=candidate_max_tokens,
        post_edit_temperature=post_edit_temperature,
        post_edit_top_p=post_edit_top_p,
        post_edit_max_tokens=post_edit_max_tokens,
        retry=retry,
    )
    records = []
    invalid_count = 0
    for index, row in frame.iterrows():
        candidates = output["candidate_generation"]["translations"][index]
        candidate_prompt = output["candidate_generation"]["prompts"][index]
        candidate_response = output["candidate_generation"]["responses"][index]
        candidate_thinking = output["candidate_generation"]["thinking"][index]
        post_edit_response = output["post_edit"]["responses"][index]
        post_edit_thinking = output["post_edit"]["thinking"][index]
        translation = output["post_edit"]["translations"][index]
        if not validate_flash_gpe_record(
            candidate_response=candidate_response,
            candidate_thinking=candidate_thinking,
            candidates=candidates,
            post_edit_response=post_edit_response,
            post_edit_thinking=post_edit_thinking,
            post_edit_translation=translation,
            max_candidates=max_candidates,
            prompt_type=prompt_type,
        ):
            invalid_count += 1
            continue
        post_edit_prompt = build_post_edit_prompt(
            row[src_lang_key],
            row[trg_lang_key],
            row[src_key],
            candidates,
            None,
        )
        record = row.to_dict()
        record.update({
            "sft_method": METHOD_ID,
            "sft_method_name": METHOD_NAME,
            "flash_gpe_prompt_type": prompt_type,
            "flash_gpe_max_candidates": max_candidates,
            "flash_gpe_candidate_count": len(candidates),
            "flash_gpe_stage1_prompt": candidate_prompt,
            "flash_gpe_stage1_thinking": candidate_thinking,
            "flash_gpe_stage1_response": candidate_response,
            "flash_gpe_candidates": candidates,
            "flash_gpe_stage2_prompt": post_edit_prompt,
            "flash_gpe_stage2_thinking": post_edit_thinking,
            "flash_gpe_stage2_response": post_edit_response,
            "flash_gpe_translation": translation,
            "flash_gpe_parser_valid": True,
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
