import fire
import pandas as pd
import itertools
import random
from typing import List, Union
from utils.config import LANG_MAP
from inference.prompts import get_GQM_prompt as get_prompt
from utils.helpers import _score_to_rank
from utils.config import candidate_identifiers
import json
import math

random.seed(114514)


def _build_score_dict(scores: List[int]) -> dict:
    return {candidate_identifiers[i]: scores[i] for i in range(len(scores))}


def _build_ground_truth(scores: List[int], prompt_type: str) -> Union[str, dict]:
    score_dict = _build_score_dict(scores)
    if prompt_type == "ranking":
        return _score_to_rank(score_dict)
    if prompt_type == "ranking_score":
        return json.dumps(score_dict)
    raise ValueError(f"Invalid prompt_type {prompt_type}")


def _sanitize_ref_text(ref_text: str, mt_texts: List[str]) -> Union[str, None]:
    if ref_text is None:
        return None
    if ref_text in mt_texts:
        return None
    return ref_text


def _build_data_item(
    src_text: str,
    mt_texts: List[str],
    src_lang: str,
    trg_lang: str,
    ref_lang: str,
    ref_text: str,
    analysis: str,
    scores: List[int],
    prompt_type: str,
    shuffle_indices: List[int],
    include_mt_texts: bool = False,
    include_reference_prompt: bool = False,
) -> dict:
    safe_ref_text = _sanitize_ref_text(ref_text, mt_texts)
    use_reference_prompt = include_reference_prompt and safe_ref_text is not None
    extra_info = {
        "src_lang": src_lang,
        "trg_lang": trg_lang,
        "ref_lang": ref_lang,
        "ref_text": ref_text,
        "reference_prompt_used": use_reference_prompt,
        "analysis": analysis,
        "shuffle_indices": shuffle_indices,
    }
    if include_mt_texts:
        extra_info["mt_texts"] = mt_texts

    return {
        "data_source": f"TowerBlocks-MT-Ranking.{prompt_type}",
        "prompt": [
            {
                "role": "user",
                "content": get_prompt(
                    src_lang,
                    trg_lang,
                    src_text,
                    mt_texts,
                    prompt_type,
                    ref_text=safe_ref_text if use_reference_prompt else None,
                    ref_lang=ref_lang if use_reference_prompt else None,
                ),
            }
        ],
        "ability": "ranking",
        "reward_model": {"ground_truth": _build_ground_truth(scores, prompt_type)},
        "extra_info": extra_info,
    }


def construct_data_item(
    src_text: str,
    mt_texts: List[str],
    src_lang: str,
    trg_lang: str,
    ref_lang: str,
    ref_text: str,
    analysis: str,
    scores: List[int],
    prompt_type: str = "ranking_score",
    shuffle_augment: int = 0,
    include_reference_prompt: bool = False,
) -> list:
    """
    Construct ranking data items, optionally with shuffle-based augmentation.
    Returns a list of data items (the original + shuffled variants).
    """
    data_items = []

    data_items.append(
        _build_data_item(
            src_text,
            mt_texts,
            src_lang,
            trg_lang,
            ref_lang,
            ref_text,
            analysis,
            scores,
            prompt_type,
            shuffle_indices=list(range(len(mt_texts))),
            include_mt_texts=True,
            include_reference_prompt=include_reference_prompt,
        )
    )

    # --- Shuffle augmentation ---
    if shuffle_augment > 0 and len(mt_texts) > 1:
        seen_orders = {tuple(range(len(mt_texts)))}  # track unique permutations
        num_generated = 0

        while num_generated < shuffle_augment:
            indices = list(range(len(mt_texts)))
            random.shuffle(indices)
            order_tuple = tuple(indices)
            if order_tuple in seen_orders:
                continue  # skip duplicates
            seen_orders.add(order_tuple)

            shuffled_mt_texts = [mt_texts[i] for i in indices]
            shuffled_scores = [scores[i] for i in indices]
            data_items.append(
                _build_data_item(
                    src_text,
                    shuffled_mt_texts,
                    src_lang,
                    trg_lang,
                    ref_lang,
                    ref_text,
                    analysis,
                    shuffled_scores,
                    prompt_type,
                    shuffle_indices=indices,
                    include_reference_prompt=include_reference_prompt,
                )
            )
            num_generated += 1

            # Stop early if we’ve exhausted all unique permutations
            if len(seen_orders) >= math.factorial(len(mt_texts)):
                break

    return data_items


def main(
    data_path: str,
    output_path: str,
    mt_key: str,
    score_key: str,
    analysis_key: str,
    prompt_type: str = "ranking_score",
    subgroup_augment: int = 0,
    shuffle_augment: int = 0,
    include_reference_prompt: bool = False,
):
    assert prompt_type in ["ranking", "ranking_score"] # TODO: support score
    df = pd.read_parquet(data_path)
    data_items = []

    for _, row in df.iterrows():
        src_text = row["src_text"]
        mt_texts = row[mt_key]
        src_lang = row["src_lang"]
        trg_lang = row["trg_lang"]
        ref_text = row["trg_text"]
        analysis = row[analysis_key]
        scores = [int(s) for s in row[score_key]]

        assert len(scores) == len(mt_texts)

        if len(src_lang) == 2:
            src_lang = LANG_MAP[src_lang]
        if len(trg_lang) == 2:
            trg_lang = LANG_MAP[trg_lang]
        ref_lang = trg_lang

        # --- Original full-sample data ---
        data_items.extend(
            construct_data_item(
                src_text,
                mt_texts,
                src_lang,
                trg_lang,
                ref_lang,
                ref_text,
                analysis,
                scores,
                prompt_type,
                shuffle_augment,
                include_reference_prompt,
            )
        )

        # --- Generate subset-based data if applicable ---
        n = len(mt_texts)
        if subgroup_augment > 0 and n > 2:
            # Generate all possible subsets of indices with size >=2 and <n
            all_subsets = []
            for r in range(2, n):
                all_subsets.extend(itertools.combinations(range(n), r))

            # Randomly sample up to subgroup_augment subsets
            random.shuffle(all_subsets)
            selected_subsets = all_subsets[:subgroup_augment]

            for subset in selected_subsets:
                subset = list(subset)
                subset_mt_texts = [mt_texts[i] for i in subset]
                subset_scores = [scores[i] for i in subset]
                data_items.extend(
                    construct_data_item(
                        src_text,
                        subset_mt_texts,
                        src_lang,
                        trg_lang,
                        ref_lang,
                        ref_text,
                        analysis,
                        subset_scores,
                        prompt_type,
                        shuffle_augment,
                        include_reference_prompt,
                    )
                )
    # Shuffle and save
    out_df = pd.DataFrame(data_items)
    out_df = out_df.sample(frac=1.0, random_state=114514)
    out_df.to_parquet(output_path, index=False)
    print(f"Total data items: {len(out_df)}")


if __name__ == "__main__":
    fire.Fire(main)
