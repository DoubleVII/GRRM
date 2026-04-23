import pandas as pd
import fire
import warnings
from inference.run_oss_GQM_with_notes import func_call


def main(
    data_path: str,
    output_path: str,
    mt_key: str,
    notes_key: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    difficulty_filter: int = 0,
    difficulty_key: str = None,
    temperature: float = 0.4,
    top_p: float = 0.7,
    retry: int = 6,
    model_path: str = "gpt-oss-20b",
    reasoning_effort: str = None,
    prompt_format: str = "score",
    add_example: bool = True,
):
    assert output_path.endswith(".parquet")

    df = pd.read_parquet(data_path)

    # Apply difficulty filter: skip rows below threshold
    active_mask = []
    notes_list = []
    for _, row in df.iterrows():
        if difficulty_filter > 0 and difficulty_key:
            difficulty = row.get(difficulty_key, 0)
            if difficulty < difficulty_filter:
                active_mask.append(False)
                notes_list.append(None)
                continue
        active_mask.append(True)
        notes_list.append(row[notes_key])

    # Initialize output with defaults for filtered rows
    n = len(df)
    all_scores = [None] * n
    all_analysis = [None] * n
    all_thinking = [None] * n

    # Process active rows only
    if any(active_mask):
        active_indices = [i for i, m in enumerate(active_mask) if m]
        active_df = df.iloc[active_indices]
        active_notes = [notes_list[i] for i in active_indices]

        out = func_call(
            src_list=active_df[src_key].tolist(),
            mt_list=active_df[mt_key].tolist(),
            notes_list=active_notes,
            src_langs=active_df[src_lang_key].tolist(),
            trg_langs=active_df[trg_lang_key].tolist(),
            temperature=temperature,
            top_p=top_p,
            retry=retry,
            model_path=model_path,
            reasoning_effort=reasoning_effort,
            prompt_format=prompt_format,
            add_example=add_example,
        )

        for j, i in enumerate(active_indices):
            all_scores[i] = out["scores"][j]
            all_analysis[i] = out["analysis"][j]
            all_thinking[i] = out["thinking"][j]

    df["oss_ranking_score"] = all_scores
    df["oss_ranking_analysis"] = all_analysis
    df["oss_ranking_thinking"] = all_thinking

    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
