import inference.run_API_GQM as run_API_GQM
import pandas as pd
import fire
import numpy as np

def process_scores(df: pd.DataFrame, text_col: str, score_col: str, analysis_col: str, model: str, src_key: str = "src_text", **kwargs) -> pd.DataFrame:
    """
    Checks for existing scores and only processes rows with NaN values if the
    score column already exists.

    Args:
        df: The main DataFrame.
        text_col: The column name for the text to be evaluated (e.g., "winner_text").
        score_col: The column name for the output score (e.g., "winner_llm_score").
        analysis_col: The column name for the output analysis (e.g., "winner_llm_analysis").
        **kwargs: Additional arguments to pass to run_API_GQM.func_call (temp, top_p).

    Returns:
        The updated DataFrame.
    """
    
    # 1. Determine which rows to process
    if score_col in df.columns:
        # Columns exist, find rows with NaN scores to re-compute
        todo_df = df[df[score_col].isna()]
        print(f"Found existing column '{score_col}'. Processing {len(todo_df)} rows with NaN values.")
    else:
        # Columns don't exist, process all rows
        todo_df = df
        print(f"Column '{score_col}' not found. Processing all {len(todo_df)} rows.")
        # Initialize new columns with pd.NA in the main df
        df[analysis_col] = pd.NA
        # the score_col should be handle separately to avoid an inhomogeneous shape error


    # 2. If there's nothing to do, return early
    if todo_df.empty:
        print(f"No rows to process for '{score_col}'.")
        return df

    # 3. Extract data for processing from the 'todo_df'
    src_texts = todo_df[src_key].tolist()
    mt_texts = todo_df[text_col].tolist()
    src_langs = todo_df["src_lang"].tolist()
    trg_langs = todo_df["trg_lang"].tolist()

    # 4. Call the LLM metric function
    assessments = run_API_GQM.func_call(
        src_texts,
        mt_texts,
        src_langs,
        trg_langs,
        temperature=kwargs.get("temperature", 0.4),
        top_p=kwargs.get("top_p", 0.9),
        model=model
    )

    
    # 5. Parse results
    assessments = assessments["response"]
    assert len(assessments) == len(todo_df), "Mismatch between assessments and number of rows to process."

    llm_scores = [ass["scores"] for ass in assessments]
    llm_analysis = [ass["analysis"] for ass in assessments]

    # 6. Update the main DataFrame at the correct indices
    # This .loc operation only updates the rows specified by todo_df.index,
    # preserving all other existing data.
    if score_col not in df.columns:
        df[score_col] = llm_scores
    else:
        df.loc[todo_df.index, score_col] = llm_scores

    df.loc[todo_df.index, analysis_col] = llm_analysis
    
    print(f"Finished processing and updating for '{score_col}'.")
    return df


def main(data_path: str, output_path: str, text_key: str, score_key: str, analysis_key: str, model: str = "gemini-2.5-pro", src_key: str = "src_text"):
    assert output_path.endswith(".parquet")
    df = pd.read_parquet(data_path)

    # Store llm_metric_ranking parameters
    llm_params = {
        "temperature": 0.4,
        "top_p": 0.9
    }

    df = process_scores(
        df=df,
        text_col=text_key,
        score_col=score_key,
        analysis_col=analysis_key,
        model=model,
        src_key=src_key,
        **llm_params
    )
    
    # --- Save Results ---
    print(f"All processing complete. Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)