import pandas as pd
import fire
import warnings
from utils.helpers import flat_list, unflat_list, repeat_text
from collections.abc import Iterable



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
    inference_type: str = "local",
    model_path: str = "gpt-oss-20b",
    reasoning_effort: str = None,
    base_url: str = "http://localhost:8000",
):
    assert output_path.endswith(".parquet")
    assert inference_type in ("local", "remote")
    if inference_type == "local":
        from inference.run_oss_post_edit import func_call
    else:
        from inference.run_oss_post_edit_remote import func_call

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
    all_pe_mt = [[] for _ in range(n)]
    all_pe_response = [[] for _ in range(n)]
    all_pe_thinking = [[] for _ in range(n)]
    all_pe_unchanged = [[] for _ in range(n)]
    for i in range(n):
        if not active_mask[i]:
            mt_row = df[mt_key].iloc[i]
            if isinstance(mt_row, Iterable):
                all_pe_mt[i] = list(mt_row)
                all_pe_response[i] = [None] * len(mt_row)
                all_pe_thinking[i] = [None] * len(mt_row)
                all_pe_unchanged[i] = [True] * len(mt_row)
            else:
                all_pe_mt[i] = [mt_row]
                all_pe_response[i] = [None]
                all_pe_thinking[i] = [None]
                all_pe_unchanged[i] = [True]

    # Process active rows only
    if any(active_mask):
        active_df = df[active_mask]
        active_notes = [n for n, m in zip(notes_list, active_mask) if m]

        mt_flat, mt_count = flat_list(active_df[mt_key].tolist())
        src_flat = repeat_text(active_df[src_key].tolist(), mt_count)
        src_lang_flat = repeat_text(active_df[src_lang_key].tolist(), mt_count)
        trg_lang_flat = repeat_text(active_df[trg_lang_key].tolist(), mt_count)
        notes_flat = repeat_text(active_notes, mt_count)

        func_call_kwargs = dict(
            src_list=src_flat,
            mt_list=mt_flat,
            notes_list=notes_flat,
            src_langs=src_lang_flat,
            trg_langs=trg_lang_flat,
            temperature=temperature,
            top_p=top_p,
            retry=retry,
        )
        if inference_type == "local":
            func_call_kwargs["model_path"] = model_path
            func_call_kwargs["reasoning_effort"] = reasoning_effort
        else:
            func_call_kwargs["model"] = model_path
            func_call_kwargs["base_url"] = base_url
            if reasoning_effort is not None:
                func_call_kwargs["reasoning_effort"] = reasoning_effort

        out = func_call(**func_call_kwargs)

        pe_mt = unflat_list(out["post_edit_mt"], mt_count)
        pe_response = unflat_list(out["response"], mt_count)
        pe_thinking = unflat_list(out["thinking"], mt_count)
        pe_unchanged = unflat_list(out["unchange"], mt_count)

        active_indices = [i for i, m in enumerate(active_mask) if m]
        for j, i in enumerate(active_indices):
            all_pe_mt[i] = pe_mt[j]
            all_pe_response[i] = pe_response[j]
            all_pe_thinking[i] = pe_thinking[j]
            all_pe_unchanged[i] = pe_unchanged[j]

    df["pe_mt"] = all_pe_mt
    df["pe_response"] = all_pe_response
    df["pe_thinking"] = all_pe_thinking
    df["pe_unchanged"] = all_pe_unchanged

    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
