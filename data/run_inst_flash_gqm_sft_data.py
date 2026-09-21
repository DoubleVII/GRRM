from pathlib import Path
import random

import fire
import pandas as pd

from inference.inst_flash_gqm_prompts import validate_candidate_count
from inference.run_inst_flash_gqm_mt import init_inst_model, run_pipeline
from inference.run_inst_flash_gqm_mt import extract_gqm_response
from inference.run_inst_flash_gpe_mt import extract_candidate_response


METHOD_ID = "inst_flash_gqm"
METHOD_NAME = "Instruct FlashGQM"


def _valid_record(candidate_response, candidates, candidate_count, gqm_response, scores, translation):
    parsed_candidates = extract_candidate_response(
        candidate_response, candidate_count, explicit_analysis=True
    )
    parsed_gqm = extract_gqm_response(
        gqm_response, len(candidates), explicit_analysis=True
    )
    return (
        parsed_candidates == candidates
        and parsed_gqm is not None
        and parsed_gqm["scores"] == scores
        and translation == candidates[max(range(len(scores)), key=scores.__getitem__)]
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
    candidate_temperature: float = 1.0,
    candidate_top_p: float = 1.0,
    candidate_top_k: int = 0,
    candidate_max_tokens: int = 8192,
    gqm_temperature: float = 1.0,
    gqm_top_p: float = 1.0,
    gqm_top_k: int = 0,
    gqm_max_tokens: int = 8192,
    retry: int = 3,
    seed: int = 114514,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Collect parser-validated Instruct FlashGQM supervision."""
    if not model_path:
        raise ValueError("model_path is required")
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    if not min_candidates <= max_candidates:
        raise ValueError("min_candidates must not exceed max_candidates")
    validate_candidate_count(min_candidates)
    validate_candidate_count(max_candidates)
    frame = pd.read_parquet(data_path)
    required = {src_key, src_lang_key, trg_lang_key}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    rng = random.Random(seed)
    candidate_counts = [
        rng.randint(min_candidates, max_candidates) for _ in range(len(frame))
    ]
    model = init_inst_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    result = run_pipeline(
        frame[src_key].tolist(), frame[src_lang_key].tolist(), frame[trg_lang_key].tolist(),
        model=model, max_candidates=max_candidates, candidate_counts=candidate_counts,
        candidate_temperature=candidate_temperature, candidate_top_p=candidate_top_p,
        candidate_top_k=candidate_top_k, candidate_max_tokens=candidate_max_tokens,
        gqm_temperature=gqm_temperature, gqm_top_p=gqm_top_p,
        gqm_top_k=gqm_top_k, gqm_max_tokens=gqm_max_tokens,
        retry=retry, enable_thinking=False,
    )
    records = []
    invalid_count = 0
    candidate = result["candidate_generation"]
    gqm = result["gqm"]
    for index, row in frame.iterrows():
        values = {
            "candidate_response": candidate["responses"][index],
            "candidates": candidate["translations"][index],
            "candidate_count": candidate_counts[index],
            "gqm_response": gqm["responses"][index],
            "scores": gqm["scores"][index],
            "translation": gqm["translations"][index],
        }
        if not values["candidates"] or values["scores"] is None or not _valid_record(**values):
            invalid_count += 1
            continue
        record = row.to_dict()
        record.update({
            "sft_method": METHOD_ID,
            "sft_method_name": METHOD_NAME,
            "flash_gqm_max_candidates": max_candidates,
            "flash_gqm_target_candidate_count": values["candidate_count"],
            "flash_gqm_candidate_count": len(values["candidates"]),
            "flash_gqm_stage1_prompt": candidate["prompts"][index],
            "flash_gqm_stage1_response": values["candidate_response"],
            "flash_gqm_candidates": values["candidates"],
            "flash_gqm_stage2_prompt": gqm["prompts"][index],
            "flash_gqm_stage2_response": values["gqm_response"],
            "flash_gqm_scores": values["scores"],
            "flash_gqm_ranking": gqm["rankings"][index],
            "flash_gqm_selected_candidate_index": gqm["selected_candidate_indices"][index],
            "flash_gqm_translation": values["translation"],
            "flash_gqm_parser_valid": True,
            "flash_gqm_protocol": "simple",
        })
        records.append(record)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(destination, index=False)
    print(f"Saved {len(records)} valid {METHOD_ID} rows to {destination}; dropped {invalid_count}")


if __name__ == "__main__":
    fire.Fire(main)
