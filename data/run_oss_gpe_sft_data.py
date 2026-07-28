from pathlib import Path

import fire
import pandas as pd

from data.oss_mt_sft_data_utils import validate_gpe_record
from inference.oss_diverse_mt_prompts import build_direct_prompt
from inference.prompts import get_oss_group_post_edit_prompt
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_group_post_edit_mt import run_pipeline, validate_sampling_n


METHOD_NAME = "Group Post-Editing"
METHOD_ID = "gpe"


def main(
    data_path: str,
    output_path: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
    sampling_n: int = 4,
    reasoning_effort: str = "medium",
    sampling_temperature: float = 0.8,
    sampling_top_p: float = 0.95,
    sampling_max_tokens: int = 4096,
    post_edit_temperature: float = 0.3,
    post_edit_top_p: float = 0.8,
    post_edit_max_tokens: int = 4096,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Collect parser-validated direct-sampling and GPE supervision."""
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    validate_sampling_n(sampling_n)
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
        sampling_n=sampling_n,
        reasoning_effort=reasoning_effort,
        sampling_temperature=sampling_temperature,
        sampling_top_p=sampling_top_p,
        sampling_max_tokens=sampling_max_tokens,
        post_edit_temperature=post_edit_temperature,
        post_edit_top_p=post_edit_top_p,
        post_edit_max_tokens=post_edit_max_tokens,
        retry=retry,
    )

    records = []
    invalid_count = 0
    for index, row in frame.iterrows():
        candidates = output["sampling"]["translations"][index]
        candidate_responses = output["sampling"]["responses"][index]
        candidate_thinking = output["sampling"]["thinking"][index]
        post_edit_translation = output["post_edit"]["translations"][index]
        post_edit_response = output["post_edit"]["responses"][index]
        post_edit_thinking = output["post_edit"]["thinking"][index]
        valid = validate_gpe_record(
            candidate_responses=candidate_responses,
            candidate_thinking=candidate_thinking,
            candidate_translations=candidates,
            post_edit_response=post_edit_response,
            post_edit_thinking=post_edit_thinking,
            post_edit_translation=post_edit_translation,
            sampling_n=sampling_n,
        )
        if not valid:
            invalid_count += 1
            continue
        stage1_prompt = build_direct_prompt(
            row[src_lang_key], row[trg_lang_key], row[src_key]
        )
        stage2_prompt = get_oss_group_post_edit_prompt(
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
            "gpe_stage1_prompts": [stage1_prompt] * sampling_n,
            "gpe_stage1_thinking": candidate_thinking,
            "gpe_stage1_responses": candidate_responses,
            "gpe_stage1_translations": candidates,
            "gpe_stage2_prompt": stage2_prompt,
            "gpe_stage2_thinking": post_edit_thinking,
            "gpe_stage2_response": post_edit_response,
            "gpe_translation": post_edit_translation,
            "gpe_parser_valid": True,
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
