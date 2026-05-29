import fire
import pandas as pd

from inference.prompts import get_group_post_edit_prompt
from utils.config import LANG_MAP


def main(
    data_path: str,
    output_path: str,
    mt_key: str = "mt_sampling_text",
    max_mt_num: int = 0,
    include_reference_info: bool = False,
):
    df = pd.read_parquet(data_path)

    data_items = []

    for _, row in df.iterrows():
        src_text = row["src_text"]
        src_lang = row["src_lang"]
        trg_lang = row["trg_lang"]
        mt_texts = row[mt_key]

        # Trim MT candidates if max_mt_num is set
        if max_mt_num > 0 and len(mt_texts) > max_mt_num:
            mt_texts = mt_texts[:max_mt_num]

        # Map 2-letter language codes to full names
        if len(src_lang) == 2:
            src_lang = LANG_MAP[src_lang]
        if len(trg_lang) == 2:
            trg_lang = LANG_MAP[trg_lang]

        prompt = get_group_post_edit_prompt(
            src_lang,
            trg_lang,
            src_text,
            mt_texts,
        )

        extra_info = {
            "src_lang": src_lang,
            "trg_lang": trg_lang,
            "mt_texts": mt_texts,
        }
        if include_reference_info:
            extra_info["ref_lang"] = trg_lang
            extra_info["ref_text"] = row["trg_text"]

        data_items.append({
            "data_source": "TowerBlocks-MT-GPE",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "group_post_edit",
            "reward_model": {"ground_truth": ""},
            "extra_info": extra_info,
        })

    out_df = pd.DataFrame(data_items)
    out_df["data_source"] = out_df["data_source"].astype("object")
    out_df["ability"] = out_df["ability"].astype("object")
    out_df = out_df.sample(frac=1.0, random_state=114514)
    out_df.to_parquet(output_path, index=False)
    print(f"Total data items: {len(out_df)}")


if __name__ == "__main__":
    fire.Fire(main)
