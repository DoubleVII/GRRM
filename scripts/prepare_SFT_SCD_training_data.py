import json
from pathlib import Path

import fire
import pandas as pd

from data.oss_mt_sft_data_utils import parse_scd_stage1_response
from inference.run_oss_diverse_mt import extract_final_translation
from inference.sft_mt_protocol import (
    build_scd_followup_prompt,
    build_scd_stage1_prompt,
    format_sft_output,
    parse_task_output,
)
from utils.config import LANG_MAP


def main(
    data_path: str,
    output_path: str,
    tokenizer_path: str = "/home/nfs06/yangs/LLM/Qwen/Qwen3-8B",
    max_length: int = 32768,
    max_samples: int = 0,
    seed: int = 114514,
    prompt_type: str = "json",
):
    """Build four-message SCD SFT conversations from OSS data."""
    from transformers import AutoTokenizer

    if prompt_type != "json":
        raise ValueError("Initial SCD SFT training supports prompt_type=json")
    frame = pd.read_parquet(data_path)
    required = {
        "src_text", "src_lang", "trg_lang",
        "scd_stage1_thinking", "scd_stage1_response", "scd_stage1_parsed",
        "scd_stage2_thinking", "scd_stage2_response", "scd_translation",
        "scd_parser_valid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    records = []
    too_long = 0
    for source_index, row in frame.iterrows():
        if not bool(row["scd_parser_valid"]):
            raise ValueError(f"Input row {source_index} is not parser-valid")
        stage1_assistant = format_sft_output(
            row["scd_stage1_thinking"], row["scd_stage1_response"]
        )
        stage1 = parse_task_output(
            stage1_assistant,
            lambda text: parse_scd_stage1_response(
                text, prompt_type="json", candidate_confidence=False
            ),
        )
        if stage1 is None:
            raise ValueError(f"SCD Stage 1 failed validation at row {source_index}")
        stored_stage1 = json.loads(row["scd_stage1_parsed"])
        if stage1["parsed"] != stored_stage1:
            raise ValueError(
                f"SCD Stage 1 parsed result differs from stored result at row {source_index}"
            )
        stage2_assistant = format_sft_output(
            row["scd_stage2_thinking"], row["scd_stage2_response"]
        )
        stage2 = parse_task_output(stage2_assistant, extract_final_translation)
        if stage2 is None or stage2["parsed"] != row["scd_translation"]:
            raise ValueError(f"SCD Stage 2 failed validation at row {source_index}")
        source_lang = LANG_MAP.get(row["src_lang"], row["src_lang"])
        target_lang = LANG_MAP.get(row["trg_lang"], row["trg_lang"])
        followup = build_scd_followup_prompt(source_lang, target_lang)
        messages = [
            {"role": "user", "content": build_scd_stage1_prompt(
                row["src_lang"], row["trg_lang"], row["src_text"]
            )},
            {"role": "assistant", "content": stage1_assistant},
            {"role": "user", "content": followup},
            {"role": "assistant", "content": stage2_assistant},
        ]
        length = len(tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        ))
        if length > max_length:
            too_long += 1
            continue
        records.append({
            "messages": messages,
            "task": "segment_level_candidate_deliberation",
            "source_index": source_index,
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "sequence_length": length,
        })

    output = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    print(f"Saved {len(output)} SCD rows to {destination}; filtered too long: {too_long}")


if __name__ == "__main__":
    fire.Fire(main)
