import pandas as pd
import fire
from inference.run_oss_prep_notes import func_call


def main(
    data_path: str,
    output_path: str,
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    temperature: float = 0.4,
    top_p: float = 0.7,
    retry: int = 6,
    model_path: str = "gpt-oss-20b",
    reasoning_effort: str = None,
):
    assert output_path.endswith(".parquet")

    df = pd.read_parquet(data_path)

    out = func_call(
        src_list=df["src_text"].tolist(),
        src_langs=df[src_lang_key].tolist(),
        trg_langs=df[trg_lang_key].tolist(),
        temperature=temperature,
        top_p=top_p,
        retry=retry,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
    )

    df["analysis"] = out["analysis"]
    df["notes"] = out["notes"]
    df["response"] = out["response"]
    df["thinking"] = out["thinking"]
    df["notes_valid"] = out["notes_valid"]
    df["difficulty"] = out["difficulty"]

    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
