import pandas as pd
import fire
import warnings
from inference.run_oss_group_post_edit import func_call


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
    temperature: float = 0.6,
    top_p: float = 0.9,
    retry: int = 6,
    model_path: str = "gpt-oss-20b",
    reasoning_effort: str = None,
):
    assert output_path.endswith(".parquet")

    df = pd.read_parquet(data_path)

    # Set notes to None for rows below difficulty threshold
    notes_list = []
    for _, row in df.iterrows():
        if difficulty_filter > 0 and difficulty_key:
            difficulty = row.get(difficulty_key, 0)
            if difficulty < difficulty_filter:
                notes_list.append(None)
                continue
        notes_list.append(row[notes_key])

    out = func_call(
        src_list=df[src_key].tolist(),
        mt_list=df[mt_key].tolist(),
        notes_list=notes_list,
        src_langs=df[src_lang_key].tolist(),
        trg_langs=df[trg_lang_key].tolist(),
        temperature=temperature,
        top_p=top_p,
        retry=retry,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
    )

    df["pe_mt"] = out["post_edit_mt"]
    df["pe_response"] = out["response"]
    df["pe_thinking"] = out["thinking"]

    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
