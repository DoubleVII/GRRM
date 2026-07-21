import fire
import pandas as pd

from inference.run_oss_GQM_post_edit_completion import func_call


def main(
    data_path: str,
    output_path: str,
    mt_key: str = "mt_sampling_text",
    gqm_analysis_key: str = "llm_ranking_analysis",
    gqm_score_key: str = "llm_ranking_score",
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    temperature: float = 0.6,
    top_p: float = 0.9,
    retry: int = 6,
    model_path: str = "gpt-oss-20b",
    reasoning_effort: str = None,
):
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")

    df = pd.read_parquet(data_path)
    required_columns = [
        src_key,
        src_lang_key,
        trg_lang_key,
        mt_key,
        gqm_analysis_key,
        gqm_score_key,
    ]
    missing_columns = [key for key in required_columns if key not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    out = func_call(
        src_list=df[src_key].tolist(),
        mt_list=df[mt_key].tolist(),
        gqm_analysis_list=df[gqm_analysis_key].tolist(),
        gqm_scores_list=df[gqm_score_key].tolist(),
        src_langs=df[src_lang_key].tolist(),
        trg_langs=df[trg_lang_key].tolist(),
        temperature=temperature,
        top_p=top_p,
        retry=retry,
        model_path=model_path,
        reasoning_effort=reasoning_effort,
    )

    df["pe_mt"] = out["post_edit_mt"]
    df["pe_analysis"] = out["post_edit_analysis"]
    df["pe_response"] = out["response"]
    df["pe_thinking"] = out["thinking"]
    df["gqm_pe_response"] = out["combined_response"]

    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
