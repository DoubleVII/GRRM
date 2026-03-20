import os
import numpy as np
import re
from typing import Optional, Dict


def get_auto_tp_size() -> int:
    tp_size = 1
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            tp_size = max(1, int(torch.cuda.device_count()))
    except Exception:
        pass
    if tp_size < 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            devs = [d.strip() for d in visible.split(",") if d.strip() not in ("", "-1")]
            tp_size = max(1, len(devs))
        else:
            tp_size = 1
    return tp_size

def get_cand_num(rank_text: str) -> int:
    rank_text = rank_text.strip()
    # 3*(x-1)+x=len(rank_text)
    # 4*x-3=len(rank_text)
    # x=(len(rank_text)+3)/4
    assert (len(rank_text) + 3) % 4 == 0
    return (len(rank_text) + 3) // 4



def flat_list(items_list: list) -> tuple[list, list]:
    item_count_list = []
    flattened_items_list = []
    for items in items_list:
        assert isinstance(items, list) or isinstance(items, np.ndarray)
        item_count_list.append(len(items))
        flattened_items_list.extend(items)

    return flattened_items_list, item_count_list

def unflat_list(flattened_items_list: list, item_count_list: list) -> list:
    assert sum(item_count_list) == len(flattened_items_list)
    unflattened_items_list = []
    start_idx = 0
    for item_count in item_count_list:
        end_idx = start_idx + item_count
        unflattened_items_list.append(flattened_items_list[start_idx:end_idx])
        start_idx = end_idx
    
    return unflattened_items_list

def repeat_text(text_list:list, repeat_count: list):
    assert len(text_list) == len(repeat_count)
    repeated_text_list = []
    for text, count in zip(text_list, repeat_count):
        repeated_text_list.extend([text] * count)
    return repeated_text_list


def _score_to_rank(d):
    # Sort by score descending, then by key ascending
    sorted_items = sorted(d.items(), key=lambda x: (-x[1], x[0]))
    
    # Group by score
    result_parts = []
    current_score = None
    current_group = []
    
    for k, v in sorted_items:
        if v != current_score:
            if current_group:
                result_parts.append(" = ".join(current_group))
            current_score = v
            current_group = [k]
        else:
            current_group.append(k)
    
    # Add the last group
    if current_group:
        result_parts.append(" = ".join(current_group))
    
    # Join groups with ' > '
    return " > ".join(result_parts)


def _ranking_to_scores(ranking_str:str) -> dict:
    # Split by '>' to separate rank groups
    groups = [grp.strip() for grp in ranking_str.split('>')]
    
    # Start from the lowest group with score 0
    scores = {}
    for rank, group in enumerate(reversed(groups)):
        # Split by '=' to get items with the same score
        items = [item.strip() for item in group.split('=')]
        for item in items:
            scores[item] = rank
    return scores


def parse_score_text(score_text: str) -> Optional[dict]:
    """
    B: 6, A: 5, C: 2
    """
    try:
        score_text = score_text.strip()
        score_text = score_text.split(",")
        score_dict = {}
        for item in score_text:
            item = item.strip()
            candidate_identifier, score = item.split(":")
            candidate_identifier = candidate_identifier.strip()
            score = int(score.strip())
            score_dict[candidate_identifier] = score
        return score_dict
    except:
        return None
    

def find_int_in_string(s: str) -> list:
    pattern = r'\d+'
    matches = re.findall(pattern, s)
    return [int(match) for match in matches]

def find_ints_in_string(s: str, expected_score_num:int=None) -> list:
    pattern = r'\d+'
    matches = re.findall(pattern, s)
    if expected_score_num is not None and len(matches) != expected_score_num:
        print(f"find {len(matches)} scores in string: {s}. Expected {expected_score_num} scores.")
        return None
    return [int(match) for match in matches]