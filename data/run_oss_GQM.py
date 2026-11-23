"""Score translation groups from Parquet with optional reference/comment evidence."""

import sys
from pathlib import Path
from typing import Optional

import fire
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.run_oss_GQM import func_call


def main(
    data_path: str,
    output_path: str,
    mt_key: str = "mt_texts",
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    ref_key: Optional[str] = None,
    comment_key: Optional[str] = None,
    difficulty_filter: int = 0,
    difficulty_key: Optional[str] = None,
    temperature: float = 0.4,
    top_p: float = 0.7,
    retry: int = 6,
    model_path: str = "gpt-oss-20b",
    reasoning_effort: Optional[str] = None,
    explicit_analysis: bool = True,
    max_new_tokens: int = 8192,
):
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    if difficulty_filter > 0 and not difficulty_key:
        raise ValueError("difficulty_key is required when difficulty_filter > 0")

    df = pd.read_parquet(data_path)
    active_indices = list(range(len(df)))
    if difficulty_filter > 0:
        active_indices = [
            i for i, value in enumerate(df[difficulty_key])
            if pd.notna(value) and value >= difficulty_filter
        ]

    columns = {
        "scores": "oss_ranking_score",
        "analysis": "oss_ranking_analysis",
        "thinking": "oss_ranking_thinking",
        "response": "oss_ranking_response",
    }
    results = {key: [None] * len(df) for key in columns}

    if active_indices:
        active_df = df.iloc[active_indices]
        evidence = {}
        for key, argument in ((ref_key, "ref_list"), (comment_key, "comment_list")):
            if key is not None:
                evidence[argument] = [
                    None if pd.isna(value) else value for value in active_df[key]
                ]

        out = func_call(
            src_list=active_df[src_key].tolist(),
            mt_list=active_df[mt_key].tolist(),
            src_langs=active_df[src_lang_key].tolist(),
            trg_langs=active_df[trg_lang_key].tolist(),
            temperature=temperature,
            top_p=top_p,
            retry=retry,
            model_path=model_path,
            reasoning_effort=reasoning_effort,
            explicit_analysis=explicit_analysis,
            max_new_tokens=max_new_tokens,
            **evidence,
        )
        for key in columns:
            if len(out[key]) != len(active_indices):
                raise ValueError(f"Output {key} length does not match active rows")
            for i, value in zip(active_indices, out[key]):
                results[key][i] = value

    for key, column in columns.items():
        df[column] = results[key]
    print(f"Processed {len(active_indices)}/{len(df)} rows. Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
