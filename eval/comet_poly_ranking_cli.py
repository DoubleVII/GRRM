import warnings
import os
from pathlib import Path
import tempfile
import subprocess
import random
import json
from utils.helpers import flat_list, unflat_list

random.seed(114514)


def get_poly_python() -> str:
    env_python = os.environ.get("COMET_POLY_RANKING_PYTHON")
    if env_python and os.path.exists(env_python):
        return env_python

    env_dir = os.environ.get("COMET_POLY_RANKING_VENV_DIR")
    if env_dir:
        candidate = os.path.join(env_dir, "bin", "python")
        if os.path.exists(candidate):
            return candidate

    warnings.warn(
        "COMET_POLY_RANKING venv not found. Falling back to system python. Set COMET_POLY_RANKING_PYTHON or COMET_POLY_RANKING_VENV_DIR to override."
    )
    return "python"


def func_call(model_path, src_list, mt_list):
    assert len(src_list) == len(
        mt_list
    ), "src_list and mt_list must have the same length"
    
    _, mt_item_count_list = flat_list(mt_list)
    # prepare comet poly input
    input_src_list = []
    mt1_list = []
    mt2_list = []
    for src, mt_group in zip(src_list, mt_list):
        assert len(mt_group) > 1, "mt_group must have at least 2 elements"
        for mt in mt_group:
            input_src_list.append(src)
            mt1_list.append(mt)
            # add a random candidate from remaining mt_group
            remaining_mt = list(mt_group).copy()
            remaining_mt.remove(mt)
            mt2_list.append(random.choice(remaining_mt))

    with (
        tempfile.NamedTemporaryFile(mode="w+t", delete=True, suffix=".json") as out_file,
        tempfile.NamedTemporaryFile(mode="w+t", delete=True, suffix=".json") as in_file,
    ):
        out_file_path = out_file.name
        in_file_path = in_file.name

        json.dump({
            "src_list": input_src_list,
            "mt_list": mt1_list,
            "mt2_list": mt2_list,
        }, in_file, ensure_ascii=False, indent=2)
        
        in_file.flush()

        in_file_path = in_file.name
        out_file_path = out_file.name

        subprocess.run(
            f"{get_poly_python()} /home/yangs/repo/COMET-poly/comet_poly/run_poly.py --model_path {model_path} --test_file {in_file_path} --output_file {out_file_path}", shell=True, check=True
        )

        out_file.seek(0) # 回到开头
        result = json.load(out_file)
        score_list = result["scores"]
        assert len(score_list) == len(input_src_list), "scores must have the same length as input_src_list"

        # unflat scores
        score_list = unflat_list(score_list, mt_item_count_list)
        assert len(score_list) == len(src_list), "score_list must have the same length as src_list"
        
    return score_list
