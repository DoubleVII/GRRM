from pathlib import Path

import fire
import pandas as pd

from data.flash_gpe_sft_data_utils import normalize_flash_gpe_row
from inference.run_oss_group_post_edit import extract_response as extract_gpe
from inference.run_oss_flash_gpe_mt import extract_candidate_response
from inference.sft_mt_protocol import (
    build_sft_flash_gpe_candidate_prompt,
    build_sft_flash_gpe_post_edit_prompt,
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
    candidate_output_path: str | None,
    flash_gpe_output_path: str | None,
) -> tuple[Path, Path]:
    base = Path(output_path)
    if base.suffix == ".parquet":
        base = base.with_suffix("")
    candidates = (
        Path(candidate_output_path)
        if candidate_output_path
        else Path(f"{base}.flash_gpe_candidates.parquet")
    )
    flash_gpe = (
        Path(flash_gpe_output_path)
        if flash_gpe_output_path
        else Path(f"{base}.flash_gpe.parquet")
    )
    if candidates == flash_gpe:
        raise ValueError("candidate and FlashGPE output paths must differ")
    return candidates, flash_gpe


def main(
    data_path: str,
    output_path: str,
    candidate_output_path: str = None,
    flash_gpe_output_path: str = None,
    tokenizer_path: str = "/home/nfs06/yangs/LLM/Qwen/Qwen3-8B",
    max_length: int = 32768,
    max_samples: int = 0,
    seed: int = 114514,
):
    """Build FlashGPE candidate-generation and post-edit SFT examples."""
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
    candidate_records = []
    flash_gpe_records = []
    too_long = {"flash_gpe_candidates": 0, "flash_gpe": 0}
    for source_index, row in frame.iterrows():
        values = normalize_flash_gpe_row(row)
        if not values["parser_valid"]:
            raise ValueError(f"Input row {source_index} is not parser-valid")
        candidate_assistant = format_sft_output(
            values["candidate_thinking"], values["candidate_response"]
        )
        parsed_candidates = parse_task_output(
            candidate_assistant,
            lambda response: extract_candidate_response(
                response,
                values["max_candidates"],
                exact_count=values["prompt_type"] == "fixed_4",
            ),
        )
        if (
            parsed_candidates is None
            or parsed_candidates["parsed"] != values["candidates"]
        ):
            raise ValueError(
                f"FlashGPE candidates failed validation at row {source_index}"
            )
        candidate_messages = [
            {
                "role": "user",
                "content": build_sft_flash_gpe_candidate_prompt(
                    row["src_lang"],
                    row["trg_lang"],
                    row["src_text"],
                    values["max_candidates"],
                    exact_count=values["prompt_type"] == "fixed_4",
                ),
            },
            {"role": "assistant", "content": candidate_assistant},
        ]
        candidate_length = _sequence_length(tokenizer, candidate_messages)
        if candidate_length > max_length:
            too_long["flash_gpe_candidates"] += 1
        else:
            candidate_records.append({
                "messages": candidate_messages,
                "task": "flash_gpe_candidates",
                "source_index": source_index,
                "src_lang": row["src_lang"],
                "trg_lang": row["trg_lang"],
                "sequence_length": candidate_length,
            })
        post_edit_assistant = format_sft_output(
            values["post_edit_thinking"], values["post_edit_response"]
        )
        parsed_post_edit = parse_task_output(post_edit_assistant, extract_gpe)
        if (
            parsed_post_edit is None
            or parsed_post_edit["parsed"] != values["translation"]
        ):
            raise ValueError(
                f"FlashGPE post-edit failed validation at row {source_index}"
            )
        post_edit_messages = [
            {
                "role": "user",
                "content": build_sft_flash_gpe_post_edit_prompt(
                    row["src_lang"],
                    row["trg_lang"],
                    row["src_text"],
                    values["candidates"],
                ),
            },
            {"role": "assistant", "content": post_edit_assistant},
        ]
        post_edit_length = _sequence_length(tokenizer, post_edit_messages)
        if post_edit_length > max_length:
            too_long["flash_gpe"] += 1
        else:
            flash_gpe_records.append({
                "messages": post_edit_messages,
                "task": "flash_gpe",
                "source_index": source_index,
                "src_lang": row["src_lang"],
                "trg_lang": row["trg_lang"],
                "sequence_length": post_edit_length,
            })
    candidate_output = pd.DataFrame(candidate_records).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    flash_gpe_output = pd.DataFrame(flash_gpe_records).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    candidate_destination, flash_gpe_destination = _output_paths(
        output_path, candidate_output_path, flash_gpe_output_path
    )
    candidate_destination.parent.mkdir(parents=True, exist_ok=True)
    flash_gpe_destination.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.to_parquet(candidate_destination, index=False)
    flash_gpe_output.to_parquet(flash_gpe_destination, index=False)
    print(
        f"Saved {len(candidate_output)} flash_gpe_candidates rows to "
        f"{candidate_destination}; filtered too long: "
        f"{too_long['flash_gpe_candidates']}"
    )
    print(
        f"Saved {len(flash_gpe_output)} flash_gpe rows to "
        f"{flash_gpe_destination}; filtered too long: "
        f"{too_long['flash_gpe']}"
    )


if __name__ == "__main__":
    fire.Fire(main)
