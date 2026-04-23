import pandas as pd
import fire
from inference.run_mt import func_call


def main(
    data_path: str,
    output_path: str,
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    sampling_n: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 4,
    model_path: str = "gpt-oss-20b",
    prompt_type: str = "codeblock-think",
    use_chat_template: bool = True,
):
    assert output_path.endswith(".parquet")

    df = pd.read_parquet(data_path)

    out = func_call(
        src_list=df["src_text"].tolist(),
        src_langs=df[src_lang_key].tolist(),
        trg_langs=df[trg_lang_key].tolist(),
        sampling_n=sampling_n,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        retry=retry,
        model_path=model_path,
        prompt_type=prompt_type,
        use_chat_template=use_chat_template,
    )

    # When sampling_n == 1, responses is a flat list; wrap each item into a list.
    # When sampling_n > 1, responses is already a nested list [num_inputs][sampling_n].
    responses = out["responses"]
    if sampling_n == 1:
        mt_sampling_text = [[r] for r in responses]
    else:
        mt_sampling_text = [list(dict.fromkeys(cands)) for cands in responses]

    df["mt_sampling_text"] = mt_sampling_text

    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
