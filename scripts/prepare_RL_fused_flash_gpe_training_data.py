from pathlib import Path
import random

import fire
import pandas as pd

from inference.oss_flash_gpe_prompts import validate_prompt_type
from inference.sft_mt_protocol import build_sft_fused_flash_gpe_prompt


DATA_SOURCE = "TowerBlocks-MT-Fused-FlashGPE"
ABILITY = "fused_flash_gpe"


def main(
    data_path: str,
    output_path: str,
    prompt_type: str = "markdown",
    min_candidates: int = 2,
    max_candidates: int = 8,
    include_reference_info: bool = True,
    max_samples: int = 0,
    repeat_times: int = 1,
    seed: int = 114514,
    testset: bool = False,
):
    """Build prompt-only Fused FlashGPE RL training data."""
    validate_prompt_type(prompt_type, max_candidates)
    if not 2 <= min_candidates <= max_candidates:
        raise ValueError("Expected 2 <= min_candidates <= max_candidates")
    if repeat_times < 1:
        raise ValueError("repeat_times must be at least 1")
    frame = pd.read_parquet(data_path)
    required = {"src_text", "src_lang", "trg_lang"}
    if include_reference_info:
        required.add("trg_text")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)

    rng = random.Random(seed)
    records = []
    for source_index, row in frame.iterrows():
        candidate_counts = []
        while len(candidate_counts) < repeat_times:
            cycle = list(range(min_candidates, max_candidates + 1))
            rng.shuffle(cycle)
            candidate_counts.extend(cycle)
        if prompt_type != "markdown":
            candidate_counts = [max_candidates] * repeat_times

        for repeat_index, candidate_count in enumerate(
            candidate_counts[:repeat_times]
        ):
            extra_info = {
                "source_index": source_index,
                "repeat_index": repeat_index,
                "src_lang": row["src_lang"],
                "trg_lang": row["trg_lang"],
                "src_text": row["src_text"],
                "prompt_type": prompt_type,
                "max_candidates": max_candidates,
                "target_candidate_count": candidate_count,
                "protocol": "markdown_headings",
            }
            if include_reference_info:
                extra_info.update({
                    "ref_lang": row["trg_lang"],
                    "ref_text": row["trg_text"],
                })
            records.append({
                "data_source": DATA_SOURCE,
                "prompt": [{
                    "role": "user",
                    "content": build_sft_fused_flash_gpe_prompt(
                        row["src_lang"],
                        row["trg_lang"],
                        row["src_text"],
                        candidate_count,
                        exact_count=prompt_type in {
                            "markdown", "fixed_4", "fixed_16"
                        },
                    ),
                }],
                "ability": ABILITY,
                "reward_model": {"ground_truth": ""} if not testset else {"ground_truth": "", 'style': 'rule'},
                "extra_info": extra_info,
            })

    output = pd.DataFrame(records).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    if not output.empty:
        output["data_source"] = output["data_source"].astype("object")
        output["ability"] = output["ability"].astype("object")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    print(f"Saved {len(output)} fused_flash_gpe RL rows to {destination}")


if __name__ == "__main__":
    fire.Fire(main)
