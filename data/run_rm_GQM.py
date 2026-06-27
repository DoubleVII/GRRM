import sys
from pathlib import Path
from typing import Optional

import fire
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference.run_rm_GQM as run_rm_GQM


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_list(value):
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def process_scores(
    df: pd.DataFrame,
    model_path: str,
    src_key: str = "src_text",
    mt_key: str = "mt_texts",
    score_key: str = "rm_score",
    response_key: str = "rm_response",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    notes_key: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 6,
    prompt_type: str = "ranking_score",
    add_example: bool = False,
) -> pd.DataFrame:
    """Run GRRM/GQM inference and persist raw responses plus parsed scores."""
    if score_key in df.columns:
        todo_mask = df[score_key].apply(_is_missing)
        todo_df = df[todo_mask]
        print(
            f"Found existing column '{score_key}'. Processing "
            f"{len(todo_df)} rows with missing scores."
        )
    else:
        todo_df = df
        df[score_key] = pd.NA
        print(f"Column '{score_key}' not found. Processing all {len(todo_df)} rows.")

    if response_key not in df.columns:
        df[response_key] = pd.NA

    if todo_df.empty:
        print("No rows to process.")
        return df

    src_list = todo_df[src_key].tolist()
    mt_list = [_as_list(value) for value in todo_df[mt_key].tolist()]
    src_langs = todo_df[src_lang_key].tolist()
    trg_langs = todo_df[trg_lang_key].tolist()

    kwargs = {}
    if notes_key is not None:
        kwargs["notes_list"] = todo_df[notes_key].tolist()

    output_dict = run_rm_GQM.func_call(
        model_path=model_path,
        src_list=src_list,
        mt_list=mt_list,
        src_langs=src_langs,
        trg_langs=trg_langs,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        retry=retry,
        prompt_type=prompt_type,
        add_example=add_example,
        **kwargs,
    )

    scores = output_dict["scores"]
    responses = output_dict["responses"]

    if len(scores) != len(todo_df) or len(responses) != len(todo_df):
        raise ValueError(
            "Mismatch between model outputs and rows to process: "
            f"got {len(scores)} scores, {len(responses)} responses, "
            f"expected {len(todo_df)}."
        )

    df.loc[todo_df.index, score_key] = pd.Series(scores, index=todo_df.index, dtype=object)
    df.loc[todo_df.index, response_key] = pd.Series(
        responses, index=todo_df.index, dtype=object
    )

    failed = sum(score is None for score in scores)
    print(
        f"Finished processing {len(todo_df)} rows for '{score_key}'. "
        f"Failed parses: {failed}."
    )
    return df


def main(
    data_path: str,
    output_path: str,
    model_path: str,
    src_key: str = "src_text",
    mt_key: str = "mt_texts",
    score_key: str = "rm_score",
    response_key: str = "rm_response",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    notes_key: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 6,
    prompt_type: str = "ranking_score",
    add_example: bool = False,
):
    assert output_path.endswith(".parquet")
    df = pd.read_parquet(data_path)

    df = process_scores(
        df=df,
        model_path=model_path,
        src_key=src_key,
        mt_key=mt_key,
        score_key=score_key,
        response_key=response_key,
        src_lang_key=src_lang_key,
        trg_lang_key=trg_lang_key,
        notes_key=notes_key,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        retry=retry,
        prompt_type=prompt_type,
        add_example=add_example,
    )

    print(f"All processing complete. Saving to {output_path}")
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
