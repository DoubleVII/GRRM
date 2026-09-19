from pathlib import Path
import random

import fire
import pandas as pd

from inference.inst_flash_gpe_prompts import validate_prompt_type
from inference.run_inst_flash_gpe_mt import (
    extract_candidate_response,
    extract_post_edit_response,
    init_inst_model,
    run_pipeline,
)


METHOD_NAME = "Instruct FlashGPE"
METHOD_ID = "inst_flash_gpe"


def _valid_record(
    *,
    candidate_response,
    candidate_thinking,
    candidates,
    candidate_count,
    post_edit_response,
    post_edit_thinking,
    translation,
    enable_thinking,
):
    """Check that both generated stages round-trip through the task parsers."""
    if enable_thinking:
        if not candidate_thinking or not post_edit_thinking:
            return False
    elif candidate_thinking is not None or post_edit_thinking is not None:
        return False
    explicit_analysis = not enable_thinking
    parsed_candidates = extract_candidate_response(
        candidate_response,
        candidate_count,
        explicit_analysis=explicit_analysis,
    )
    parsed_translation = extract_post_edit_response(
        post_edit_response,
        explicit_analysis=explicit_analysis,
    )
    return (
        parsed_candidates == candidates
        and parsed_translation is not None
        and parsed_translation == translation
    )


def main(
    data_path: str,
    output_path: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    model_path: str = None,
    max_samples: int = 0,
    min_candidates: int = 2,
    max_candidates: int = 8,
    prompt_type: str = "markdown",
    candidate_temperature: float = 1.0,
    candidate_top_p: float = 1.0,
    candidate_top_k: int = 0,
    candidate_presence_penalty: float = 0.0,
    candidate_repetition_penalty: float = 1.0,
    candidate_max_tokens: int = 8192,
    post_edit_temperature: float = 1.0,
    post_edit_top_p: float = 1.0,
    post_edit_top_k: int = 0,
    post_edit_presence_penalty: float = 0.0,
    post_edit_repetition_penalty: float = 1.0,
    post_edit_max_tokens: int = 8192,
    retry: int = 3,
    seed: int = 114514,
    enable_thinking: bool = True,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Collect parser-validated Instruct FlashGPE supervision."""
    if not model_path:
        raise ValueError("model_path is required")
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    if not 2 <= min_candidates <= max_candidates:
        raise ValueError("Expected 2 <= min_candidates <= max_candidates")
    validate_prompt_type(prompt_type, max_candidates)
    if retry < 0:
        raise ValueError("retry must be non-negative")

    frame = pd.read_parquet(data_path)
    required = {src_key, src_lang_key, trg_lang_key}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    print("data size:", len(frame))

    rng = random.Random(seed)
    candidate_counts = [
        rng.randint(min_candidates, max_candidates) for _ in range(len(frame))
    ]
    model = init_inst_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    output = run_pipeline(
        frame[src_key].tolist(),
        frame[src_lang_key].tolist(),
        frame[trg_lang_key].tolist(),
        model=model,
        max_candidates=max_candidates,
        candidate_counts=candidate_counts,
        candidate_temperature=candidate_temperature,
        candidate_top_p=candidate_top_p,
        candidate_top_k=candidate_top_k,
        candidate_presence_penalty=candidate_presence_penalty,
        candidate_repetition_penalty=candidate_repetition_penalty,
        candidate_max_tokens=candidate_max_tokens,
        post_edit_temperature=post_edit_temperature,
        post_edit_top_p=post_edit_top_p,
        post_edit_top_k=post_edit_top_k,
        post_edit_presence_penalty=post_edit_presence_penalty,
        post_edit_repetition_penalty=post_edit_repetition_penalty,
        post_edit_max_tokens=post_edit_max_tokens,
        retry=retry,
        enable_thinking=enable_thinking,
    )

    records = []
    invalid_count = 0
    for index, row in frame.iterrows():
        candidate = output["candidate_generation"]
        post_edit = output["post_edit"]
        candidates = candidate["translations"][index]
        candidate_count = candidate_counts[index]
        translation = post_edit["translations"][index]
        valid = _valid_record(
            candidate_response=candidate["responses"][index],
            candidate_thinking=candidate["thinking"][index],
            candidates=candidates,
            candidate_count=candidate_count,
            post_edit_response=post_edit["responses"][index],
            post_edit_thinking=post_edit["thinking"][index],
            translation=translation,
            enable_thinking=enable_thinking,
        )
        if not valid:
            invalid_count += 1
            continue
        record = row.to_dict()
        record.update({
            "sft_method": METHOD_ID,
            "sft_method_name": METHOD_NAME,
            "flash_gpe_prompt_type": prompt_type,
            "flash_gpe_max_candidates": max_candidates,
            "flash_gpe_target_candidate_count": candidate_count,
            "flash_gpe_candidate_count": len(candidates),
            "flash_gpe_protocol": "markdown_headings",
            "flash_gpe_stage1_prompt": candidate["prompts"][index],
            "flash_gpe_stage1_thinking": candidate["thinking"][index],
            "flash_gpe_stage1_response": candidate["responses"][index],
            "flash_gpe_candidates": candidates,
            "flash_gpe_stage2_prompt": post_edit["prompts"][index],
            "flash_gpe_stage2_thinking": post_edit["thinking"][index],
            "flash_gpe_stage2_response": post_edit["responses"][index],
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
