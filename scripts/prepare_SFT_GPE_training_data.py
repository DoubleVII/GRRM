from pathlib import Path

import fire
import pandas as pd

from inference.run_oss_diverse_mt import extract_final_translation
from inference.run_oss_group_post_edit import extract_response as extract_gpe
from inference.sft_mt_protocol import (
    build_sft_direct_prompt,
    build_sft_gpe_prompt,
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


def _output_paths(
    output_path: str,
    direct_output_path: str | None,
    post_edit_output_path: str | None,
) -> tuple[Path, Path]:
    base = Path(output_path)
    if base.suffix == ".parquet":
        base = base.with_suffix("")
    direct = Path(direct_output_path) if direct_output_path else Path(
        f"{base}.direct_mt.parquet"
    )
    post_edit = Path(post_edit_output_path) if post_edit_output_path else Path(
        f"{base}.group_post_edit.parquet"
    )
    if direct == post_edit:
        raise ValueError("direct and group-post-edit output paths must differ")
    return direct, post_edit


def main(
    data_path: str,
    output_path: str,
    direct_output_path: str = None,
    post_edit_output_path: str = None,
    tokenizer_path: str = "/home/nfs06/yangs/LLM/Qwen/Qwen3-8B",
    max_length: int = 32768,
    max_samples: int = 0,
    seed: int = 114514,
):
    """Build direct-MT and group-post-edit SFT examples from OSS data."""
    from transformers import AutoTokenizer

    frame = pd.read_parquet(data_path)
    required = {
        "src_text", "src_lang", "trg_lang",
        "gpe_stage1_thinking", "gpe_stage1_responses",
        "gpe_stage1_translations",
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

    direct_records = []
    post_edit_records = []
    too_long = {"direct_mt": 0, "group_post_edit": 0}
    for source_index, row in frame.iterrows():
        if not bool(row["gpe_parser_valid"]):
            raise ValueError(f"Input row {source_index} is not parser-valid")
        direct_values = list(zip(
            row["gpe_stage1_thinking"], row["gpe_stage1_responses"],
            row["gpe_stage1_translations"],
        ))
        if len(direct_values) != 4:
            raise ValueError(
                f"Input row {source_index} has {len(direct_values)} direct candidates; expected 4"
            )
        direct_prompt = build_sft_direct_prompt(
            row["src_lang"], row["trg_lang"], row["src_text"]
        )
        for candidate_index, (thinking, response, translation) in enumerate(direct_values):
            assistant = format_sft_output(thinking, response)
            parsed = parse_task_output(assistant, extract_final_translation)
            if parsed is None or parsed["parsed"] != translation:
                raise ValueError(
                    f"Direct response failed round-trip validation at row {source_index}, candidate {candidate_index}"
                )
            messages = [
                {"role": "user", "content": direct_prompt},
                {"role": "assistant", "content": assistant},
            ]
            length = _sequence_length(tokenizer, messages)
            if length > max_length:
                too_long["direct_mt"] += 1
                continue
            direct_records.append({
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
            {"role": "user", "content": build_sft_gpe_prompt(
                row["src_lang"], row["trg_lang"], row["src_text"],
                list(row["gpe_stage1_translations"]),
            )},
            {"role": "assistant", "content": assistant},
        ]
        length = _sequence_length(tokenizer, messages)
        if length > max_length:
            too_long["group_post_edit"] += 1
        else:
            post_edit_records.append({
                "messages": messages,
                "task": "group_post_edit",
                "source_index": source_index,
                "candidate_index": -1,
                "src_lang": row["src_lang"],
                "trg_lang": row["trg_lang"],
                "sequence_length": length,
            })

    direct_output = pd.DataFrame(direct_records).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    post_edit_output = pd.DataFrame(post_edit_records).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    direct_destination, post_edit_destination = _output_paths(
        output_path, direct_output_path, post_edit_output_path
    )
    direct_destination.parent.mkdir(parents=True, exist_ok=True)
    post_edit_destination.parent.mkdir(parents=True, exist_ok=True)
    direct_output.to_parquet(direct_destination, index=False)
    post_edit_output.to_parquet(post_edit_destination, index=False)
    print(
        f"Saved {len(direct_output)} direct_mt rows to {direct_destination}; "
        f"filtered too long: {too_long['direct_mt']}"
    )
    print(
        f"Saved {len(post_edit_output)} group_post_edit rows to "
        f"{post_edit_destination}; filtered too long: "
        f"{too_long['group_post_edit']}"
    )


if __name__ == "__main__":
    fire.Fire(main)
