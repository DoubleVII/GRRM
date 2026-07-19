import fire
import pandas as pd

from inference.run_prep_notes import func_call


def main(
    data_path: str,
    output_path: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    retry: int = 6,
    model_path: str = "Qwen/Qwen3-8B",
    use_simple_prompt: bool = True,
):
    assert output_path.endswith(".parquet")

    df = pd.read_parquet(data_path)

    out = func_call(
        src_list=df[src_key].tolist(),
        src_langs=df[src_lang_key].tolist(),
        trg_langs=df[trg_lang_key].tolist(),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        retry=retry,
        model_path=model_path,
        use_simple_prompt=use_simple_prompt,
    )

    df["analysis"] = out["analysis"]
    df["notes"] = out["notes"]
    df["response"] = out["response"]
    df["notes_valid"] = out["notes_valid"]
    df["difficulty"] = out["difficulty"]

    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
