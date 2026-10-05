import json
from pathlib import Path

import fire
import pandas as pd

from inference.run_oss_key_points import run_key_points_stage
from inference.run_oss_SQM import init_oss_model


def main(
    data_path: str,
    output_path: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
    reasoning_effort: str = "medium",
    temperature: float = 0.3,
    top_p: float = 0.8,
    max_tokens: int = 4096,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
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
    output = run_key_points_stage(
        frame[src_key].tolist(), frame[src_lang_key].tolist(), frame[trg_lang_key].tolist(),
        model=model, model_path=model_path, reasoning_effort=reasoning_effort,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, retry=retry,
    )
    records = []
    failed = 0
    for index, row in frame.iterrows():
        parsed = output["key_points"][index]
        record = row.to_dict()
        if parsed is None:
            failed += 1
            record.update({
                "key_points": None,
                "key_points_prompt": output["prompts"][index],
                "key_points_thinking": output["thinking"][index],
                "key_points_response": output["responses"][index],
                "key_points_parser_valid": False,
            })
            records.append(record)
            continue
        record.update({
            "key_points": json.dumps(parsed["key_points"], ensure_ascii=False),
            "key_points_prompt": output["prompts"][index],
            "key_points_thinking": output["thinking"][index],
            "key_points_response": output["responses"][index],
            "key_points_parser_valid": True,
        })
        records.append(record)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_columns = list(frame.columns) + [
        "key_points", "key_points_prompt", "key_points_thinking",
        "key_points_response", "key_points_parser_valid",
    ]
    pd.DataFrame(records, columns=output_columns).to_parquet(destination, index=False)
    print(f"Saved {len(records)} key-point rows to {destination}; {failed} rows have parser-invalid results")


if __name__ == "__main__":
    fire.Fire(main)
