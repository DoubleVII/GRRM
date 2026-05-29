import random

import fire
import pandas as pd

from inference.run_mt import get_prompt
from utils.config import TOWER_LANGS


RANDOM_STATE = 114514
DATA_SOURCE = "towerblocks"
PROMPT_TEMPLATE = "codeblock-think"

random.seed(RANDOM_STATE)


def build_data_item(row, src_lang, trg_lang, testset=False, include_reference_info=True):
    src_text = row["src_text"]
    ref_text = row["trg_text"]
    ref_lang = row["trg_lang"]
    prompt = get_prompt(PROMPT_TEMPLATE, src_lang, trg_lang, src_text)
    data_item = {
        "prompt": [
            {"role": "user", "content": prompt}
        ],
        "ability": "translation",
        "data_source": DATA_SOURCE,
        "extra_info": {
            "src_lang": src_lang,
            "trg_lang": trg_lang,
            "src_text": src_text,
        }
    }
    if include_reference_info:
        data_item["extra_info"]["ref_lang"] = ref_lang
        data_item["extra_info"]["ref_text"] = ref_text

    if testset:
        data_item["reward_model"] = {"style": "rule", "ground_truth": ref_text}

    return data_item


def maybe_shuffle(df, testset=False):
    if testset:
        return df
    return df.sample(frac=1, random_state=RANDOM_STATE)


def maybe_sample(df, sample_size=None):
    if sample_size is None:
        return df
    sample_size = int(sample_size)
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    sample_size = min(sample_size, len(df))
    return df.sample(n=sample_size, random_state=RANDOM_STATE)


def extra_info_frame(df):
    if df.empty or "extra_info" not in df.columns:
        return pd.DataFrame()
    return pd.json_normalize(df["extra_info"])


def print_distribution(series, title, top_k=None):
    counts = series.value_counts(dropna=False)
    if top_k is not None:
        counts = counts.head(top_k)

    print(f"{title}:")
    if counts.empty:
        print("  <empty>")
        return

    for value, count in counts.items():
        print(f"  {value}: {count}")


def print_data_stats(df, name, top_k_pairs=20):
    extra_info_df = extra_info_frame(df)

    print(f"[{name}] total rows: {len(df)}")
    if extra_info_df.empty:
        return

    for lang_column in ["src_lang", "trg_lang", "ref_lang"]:
        if lang_column in extra_info_df:
            print_distribution(extra_info_df[lang_column], f"[{name}] {lang_column} distribution")

    if {"src_lang", "trg_lang"}.issubset(extra_info_df.columns):
        direction = extra_info_df["src_lang"] + "->" + extra_info_df["trg_lang"]
        print_distribution(
            direction,
            f"[{name}] translation direction distribution (top {top_k_pairs})",
            top_k=top_k_pairs,
        )

    if {"trg_lang", "ref_lang"}.issubset(extra_info_df.columns):
        ref_mismatch_count = (extra_info_df["trg_lang"] != extra_info_df["ref_lang"]).sum()
        print(f"[{name}] rows where trg_lang != ref_lang: {ref_mismatch_count}")


def run_prepare(df, testset=False, include_reference_info=True):
    output = []

    for _, row in df.iterrows():
        src_lang = row["src_lang"]
        trg_lang = row["trg_lang"]
        output.append(build_data_item(row, src_lang, trg_lang, testset, include_reference_info))

    out_df = pd.DataFrame(output)
    return maybe_shuffle(out_df, testset)


def run_prepare_towerx(df, testset=False, trg_lang_num=1, include_reference_info=True):
    """
    Prepare cross-lingual augmented x2x data.
    """
    output = []

    for _, row in df.iterrows():
        src_lang = row["src_lang"]
        trg_lang = row["trg_lang"]
        langs_candidate = TOWER_LANGS.copy()
        langs_candidate.remove(src_lang)
        langs_candidate.remove(trg_lang)
        sample_size = min(trg_lang_num, len(langs_candidate))
        trg_langs = random.sample(langs_candidate, sample_size)

        for trg_lang in trg_langs:
            output.append(build_data_item(row, src_lang, trg_lang, testset, include_reference_info))

    out_df = pd.DataFrame(output)
    return maybe_shuffle(out_df, testset)


def construct_tower(data_path, output_path, testset=False, include_reference_info=True):
    df = pd.read_parquet(data_path)
    df = run_prepare(df, testset, include_reference_info)
    print_data_stats(df, "tower")
    df.to_parquet(output_path, index=False)


def construct_towerx(
    data_path,
    output_path,
    testset=False,
    testset_sample_size=None,
    trg_lang_num=1,
    include_reference_info=False,
):
    """
    Build TowerX data.

    Train split: original Tower data + x2x augmented data.
    Test split: sampled from x2x augmented data only.
    """
    df = pd.read_parquet(data_path)
    x2x_df = run_prepare_towerx(
        df,
        testset,
        trg_lang_num=trg_lang_num,
        include_reference_info=include_reference_info,
    )

    if testset:
        output_df = maybe_sample(x2x_df, testset_sample_size)
    else:
        en_df = run_prepare(df, testset, include_reference_info)
        output_df = pd.concat([en_df, x2x_df], axis=0)
        output_df = output_df.sample(frac=1, random_state=RANDOM_STATE)
        print(f"[towerx] original rows: {len(en_df)}")
        print(f"[towerx] x2x rows: {len(x2x_df)}")

    print_data_stats(output_df, "towerx")
    output_df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    fire.Fire({
        "construct_tower": construct_tower,
        "construct_towerx": construct_towerx,
    })
