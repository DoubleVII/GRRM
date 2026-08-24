from pathlib import Path

import fire
import pandas as pd

from data.flash_gpe_sft_data_utils import normalize_flash_gpe_row
from inference.run_oss_flash_gpe_mt import extract_candidate_response
from inference.run_oss_group_post_edit import extract_response as extract_gpe
from inference.sft_mt_protocol import (
    build_sft_fused_flash_gpe_prompt,
    format_fused_sft_output,
    parse_fused_task_output,
)


def _sequence_length(tokenizer, messages: list[dict]) -> int:
    return len(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
    ))


def main(
    data_path: str,
    output_path: str,
    tokenizer_path: str = "/home/nfs06/yangs/LLM/Qwen/Qwen3-8B",
    max_length: int = 32768,
    max_samples: int = 0,
    seed: int = 114514,
):
    """Build one-pass Fused FlashGPE SFT examples."""
    from transformers import AutoTokenizer

    frame = pd.read_parquet(data_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )

    records = []
    too_long = 0
    for source_index, row in frame.iterrows():
        values = normalize_flash_gpe_row(row)
        if not values["parser_valid"]:
            raise ValueError(f"Input row {source_index} is not parser-valid")
        exact_count = values["prompt_type"] in {"markdown", "fixed_4", "fixed_16"}
        assistant = format_fused_sft_output(
            values["candidate_thinking"],
            values["candidate_response"],
            values["post_edit_thinking"],
            values["post_edit_response"],
        )
        parsed = parse_fused_task_output(
            assistant,
            lambda response: extract_candidate_response(
                response,
                values["max_candidates"],
                exact_count=exact_count,
            ),
            extract_gpe,
        )
        if (
            parsed is None
            or parsed["candidates"] != values["candidates"]
            or parsed["parsed"] != values["translation"]
        ):
            raise ValueError(
                f"Fused FlashGPE output failed validation at row {source_index}"
            )
        messages = [
            {
                "role": "user",
                "content": build_sft_fused_flash_gpe_prompt(
                    row["src_lang"],
                    row["trg_lang"],
                    row["src_text"],
                    values["max_candidates"],
                    exact_count=exact_count,
                ),
            },
            {"role": "assistant", "content": assistant},
        ]
        length = _sequence_length(tokenizer, messages)
        if length > max_length:
            too_long += 1
            continue
        records.append({
            "messages": messages,
            "task": "fused_flash_gpe",
            "source_index": source_index,
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "sequence_length": length,
        })

    output = pd.DataFrame(records).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    print(
        f"Saved {len(output)} fused_flash_gpe rows to {destination}; "
        f"filtered too long: {too_long}"
    )


if __name__ == "__main__":
    fire.Fire(main)
