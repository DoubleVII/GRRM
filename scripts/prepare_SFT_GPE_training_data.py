from pathlib import Path

import fire
import pandas as pd

from inference.run_oss_diverse_mt import extract_final_translation
from inference.run_oss_group_post_edit import extract_response as extract_gpe
from inference.sft_mt_protocol import (
    add_output_instruction,
    format_sft_output,
    parse_task_output,
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
    """Build direct-MT and group-post-edit SFT examples from OSS data."""
    from transformers import AutoTokenizer

    frame = pd.read_parquet(data_path)
    required = {
        "src_text", "src_lang", "trg_lang", "gpe_stage1_prompts",
        "gpe_stage1_thinking", "gpe_stage1_responses",
        "gpe_stage1_translations", "gpe_stage2_prompt",
        "gpe_stage2_thinking", "gpe_stage2_response", "gpe_translation",
        "gpe_parser_valid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    records = []
    too_long = {"direct_mt": 0, "group_post_edit": 0}
    for source_index, row in frame.iterrows():
        if not bool(row["gpe_parser_valid"]):
            raise ValueError(f"Input row {source_index} is not parser-valid")
        direct_values = list(zip(
            row["gpe_stage1_prompts"], row["gpe_stage1_thinking"],
            row["gpe_stage1_responses"], row["gpe_stage1_translations"],
        ))
        if len(direct_values) != 4:
            raise ValueError(
                f"Input row {source_index} has {len(direct_values)} direct candidates; expected 4"
            )
        for candidate_index, (prompt, thinking, response, translation) in enumerate(direct_values):
            assistant = format_sft_output(thinking, response)
            parsed = parse_task_output(assistant, extract_final_translation)
            if parsed is None or parsed["parsed"] != translation:
                raise ValueError(
                    f"Direct response failed round-trip validation at row {source_index}, candidate {candidate_index}"
                )
            messages = [
                {"role": "user", "content": add_output_instruction(prompt)},
                {"role": "assistant", "content": assistant},
            ]
            length = _sequence_length(tokenizer, messages)
            if length > max_length:
                too_long["direct_mt"] += 1
                continue
            records.append({
                "messages": messages,
                "task": "direct_mt",
                "source_index": source_index,
                "candidate_index": candidate_index,
                "src_lang": row["src_lang"],
                "trg_lang": row["trg_lang"],
                "sequence_length": length,
            })

        assistant = format_sft_output(
            row["gpe_stage2_thinking"], row["gpe_stage2_response"]
        )
        parsed = parse_task_output(assistant, extract_gpe)
        if parsed is None or parsed["parsed"] != row["gpe_translation"]:
            raise ValueError(
                f"GPE response failed round-trip validation at row {source_index}"
            )
        messages = [
            {"role": "user", "content": add_output_instruction(row["gpe_stage2_prompt"])},
            {"role": "assistant", "content": assistant},
        ]
        length = _sequence_length(tokenizer, messages)
        if length > max_length:
            too_long["group_post_edit"] += 1
        else:
            records.append({
                "messages": messages,
                "task": "group_post_edit",
                "source_index": source_index,
                "candidate_index": -1,
                "src_lang": row["src_lang"],
                "trg_lang": row["trg_lang"],
                "sequence_length": length,
            })

    output = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    counts = output["task"].value_counts().to_dict()
    print(f"Saved {len(output)} rows to {destination}: {counts}; filtered too long: {too_long}")


if __name__ == "__main__":
    fire.Fire(main)
