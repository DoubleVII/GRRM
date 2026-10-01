from pathlib import Path

import fire
import pandas as pd

from data.oss_mt_sft_data_utils import nonempty_text
from inference.oss_diverse_mt_prompts import build_direct_prompt
from inference.run_oss_SQM import init_oss_model
from inference.run_oss_direct_mt import run_direct_stage
from inference.run_oss_diverse_mt import extract_final_translation


METHOD_NAME = "Direct Translation"
METHOD_ID = "direct_mt"


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
    """Collect parser-validated direct translation supervision from gpt-oss."""
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
    output = run_direct_stage(
        frame[src_key].tolist(),
        frame[src_lang_key].tolist(),
        frame[trg_lang_key].tolist(),
        model=model,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        retry=retry,
    )

    records = []
    invalid_count = 0
    for index, row in frame.iterrows():
        response = output["responses"][index]
        thinking = output["thinking"][index]
        translation = output["translations"][index]
        parsed = extract_final_translation(response) if nonempty_text(response) else None
        valid = (
            nonempty_text(thinking)
            and nonempty_text(parsed)
            and parsed == translation
        )
        if not valid:
            invalid_count += 1
            continue
        prompt = build_direct_prompt(
            row[src_lang_key], row[trg_lang_key], row[src_key]
        )
        record = row.to_dict()
        record.update({
            "sft_method": METHOD_ID,
            "sft_method_name": METHOD_NAME,
            "direct_mt_prompt": prompt,
            "direct_mt_thinking": thinking,
            "direct_mt_response": response,
            "direct_mt_translation": translation,
            "direct_mt_parser_valid": True,
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
