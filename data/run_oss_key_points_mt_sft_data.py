import json
from pathlib import Path

import pandas as pd

from inference.run_oss_SQM import init_oss_model
from inference.run_oss_key_points import parse_key_points_response
from inference.run_oss_key_points_mt import extract_final_translation, run_pipeline


METHOD_ID = "key_points_mt"
METHOD_NAME = "KeyPointsMT"
OUTPUT_COLUMNS = ["sft_method", "sft_method_name", "key_points_mt_key_points",
                  "key_points_mt_stage1_prompt", "key_points_mt_stage1_thinking",
                  "key_points_mt_stage1_response", "key_points_mt_stage2_prompt",
                  "key_points_mt_stage2_thinking", "key_points_mt_stage2_response",
                  "key_points_mt_translation", "key_points_mt_parser_valid"]


def validate_record(output: dict, index: int, source: str) -> bool:
    first, second = output["key_points"], output["convergent"]
    points = parse_key_points_response(first["responses"][index], source)
    translation = extract_final_translation(second["responses"][index])
    return (
        points is not None and points == first["key_points"][index]
        and translation is not None and translation == second["translations"][index]
        and all(isinstance(value, str) and bool(value.strip()) for value in (
            first["prompts"][index], second["prompts"][index],
            first["thinking"][index], second["thinking"][index],
        ))
    )


def main(
    data_path: str,
    output_path: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
    reasoning_effort: str = "medium",
    key_points_temperature: float = 1.0,
    key_points_top_p: float = 1.0,
    stage1_max_tokens: int = 8192,
    final_temperature: float = 1.0,
    final_top_p: float = 1.0,
    final_max_tokens: int = 8192,
    retry: int = 3,
    allow_key_points_failure: bool = False,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Collect complete, parser-validated two-stage reasoning supervision."""
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    frame = pd.read_parquet(data_path)
    missing = sorted({src_key, src_lang_key, trg_lang_key} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    print("data size:", len(frame))
    model = init_oss_model(model_path, gpu_memory_utilization=gpu_memory_utilization,
                           max_model_len=max_model_len) if len(frame) else None
    output = run_pipeline(
        frame[src_key].tolist(), frame[src_lang_key].tolist(), frame[trg_lang_key].tolist(),
        model=model, model_path=model_path, reasoning_effort=reasoning_effort,
        key_points_temperature=key_points_temperature, key_points_top_p=key_points_top_p,
        stage1_max_tokens=stage1_max_tokens, final_temperature=final_temperature,
        final_top_p=final_top_p, final_max_tokens=final_max_tokens, retry=retry,
        allow_key_points_failure=allow_key_points_failure,
    )
    first, second = output["key_points"], output["convergent"]
    records = []
    for i, row in frame.iterrows():
        if not validate_record(output, i, row[src_key]):
            continue
        record = row.to_dict()
        record.update({
            "sft_method": METHOD_ID, "sft_method_name": METHOD_NAME,
            "key_points_mt_key_points": json.dumps(first["key_points"][i], ensure_ascii=False),
            "key_points_mt_stage1_prompt": first["prompts"][i],
            "key_points_mt_stage1_thinking": first["thinking"][i],
            "key_points_mt_stage1_response": first["responses"][i],
            "key_points_mt_stage2_prompt": second["prompts"][i],
            "key_points_mt_stage2_thinking": second["thinking"][i],
            "key_points_mt_stage2_response": second["responses"][i],
            "key_points_mt_translation": second["translations"][i],
            "key_points_mt_parser_valid": True,
        })
        records.append(record)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys([*frame.columns, *OUTPUT_COLUMNS]))
    pd.DataFrame(records, columns=columns).to_parquet(destination, index=False)
    print(f"Saved {len(records)} valid {METHOD_ID} rows to {destination}; "
          f"dropped {len(frame) - len(records)} invalid SFT rows")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
