import pandas as pd
from pathlib import Path

from typing import Optional, Dict, Any, List, Iterable
from collections import defaultdict
from utils.helpers import _score_to_rank
import wandb

from utils.config import RANKING_TEST_DATA_META_INFO
import utils.reward_utils as reward_utils
from utils.config import candidate_identifiers
import json
import numpy as np



def log_results_to_wandb(
    datasets_metric_results: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    scale: float = 100,
):
    project_name = "rm-eval"
    wandb.init(
        project=project_name,
        name=f"{config['model_name']}",
        config=config,
    )
    metrics = list(next(iter(datasets_metric_results.values())).keys())
    columns = ["dataset_name"] + metrics
    metric_data = []
    for dataset_name, metric_results in datasets_metric_results.items():
        metric_data.append([])
        metric_data[-1].append(dataset_name)
        for metric_name in metrics:
            if metric_name not in metric_results:
                raise ValueError(f"metric_name {metric_name} not in metric_results")
            metric_data[-1].append(metric_results[metric_name] * scale)

    table = wandb.Table(columns=columns, data=metric_data)
    wandb.log({f"ranking_acc_results": table})


def run_rm_SQM_inference(
    df: pd.DataFrame,
    model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    runs: int,
    model=None,
    tokenizer=None,
):
    import inference.run_rm_SQM as run_rm_SQM
    from utils.helpers import flat_list, unflat_list, repeat_text

    src_list = df["src_text"].tolist()
    mt_list = df["mt_texts"].tolist()
    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()

    mt_list, mt_item_count_list = flat_list(mt_list)
    src_list = repeat_text(src_list, mt_item_count_list)
    src_langs = repeat_text(src_langs, mt_item_count_list)
    trg_langs = repeat_text(trg_langs, mt_item_count_list)

    if model is None or tokenizer is None:
        from inference.run_rm_SQM import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(model_path)

    score_for_runs = []
    for _ in range(runs):
        output_dict = run_rm_SQM.func_call(
            model_path,
            src_list,
            mt_list,
            src_langs,
            trg_langs,
            temperature,
            top_p,
            max_new_tokens,
            model=model,
            tokenizer=tokenizer,
        )
        score_list = output_dict["scores"]
        score_list = unflat_list(score_list, mt_item_count_list)
        if len(score_list) != len(df):
            raise ValueError(
                f"score_list must have the same length as src_list, but got {len(score_list)} and {len(df)}"
            )

        score_for_runs.append(score_list)

    return score_for_runs


def run_drm_inference(
    df: pd.DataFrame,
    model_path: str,
    runs: int,
    model=None,
    tokenizer=None,
):
    import inference.run_drm as run_drm
    from utils.helpers import flat_list, unflat_list, repeat_text

    src_list = df["src_text"].tolist()
    mt_list = df["mt_texts"].tolist()
    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()

    mt_list, mt_item_count_list = flat_list(mt_list)
    src_list = repeat_text(src_list, mt_item_count_list)
    src_langs = repeat_text(src_langs, mt_item_count_list)
    trg_langs = repeat_text(trg_langs, mt_item_count_list)

    if model is None or tokenizer is None:
        from inference.run_drm import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(model_path)

    score_for_runs = []
    for _ in range(runs):
        output_dict = run_drm.func_call(
            src_list,
            mt_list,
            src_langs,
            trg_langs,
            model=model,
            tokenizer=tokenizer,
        )
        score_list = output_dict["scores"]
        score_list = unflat_list(score_list, mt_item_count_list)
        if len(score_list) != len(df):
            raise ValueError(
                f"score_list must have the same length as src_list, but got {len(score_list)} and {len(df)}"
            )

        score_for_runs.append(score_list)

    return score_for_runs


def run_rm_GQM_inference(
    df: pd.DataFrame,
    model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    prompt_type: str,
    runs: int,
    add_prompt_example: bool = False,
    model=None,
    tokenizer=None,
):
    import inference.run_rm_GQM as run_rm_GQM
    
    src_list = df["src_text"].tolist()
    mt_list = df["mt_texts"].tolist()
    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()

    if model is None or tokenizer is None:
        from inference.run_rm_GQM import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(model_path)

    score_for_runs = []
    for _ in range(runs):
        output_dict = run_rm_GQM.func_call(
            model_path,
            src_list,
            mt_list,
            src_langs,
            trg_langs,
            temperature,
            top_p,
            max_new_tokens,
            prompt_type=prompt_type,
            add_example=add_prompt_example,
            model=model,
            tokenizer=tokenizer,
        )
        score_list = output_dict["scores"]
        if len(score_list) != len(df):
            raise ValueError(
                f"score_list must have the same length as src_list, but got {len(score_list)} and {len(df)}"
            )

        score_for_runs.append(score_list)

    return score_for_runs


def run_eval(
    df: pd.DataFrame,
    score_for_runs: list[list[int]],
):
    ref_scores_list = df["scores"].tolist()

    metric_output = defaultdict(list)
    reward_fn = reward_utils.ranking_reward_fn_no_cot

    for score_list in score_for_runs:
        for pred_scores, ref_scores in zip(score_list, ref_scores_list):
            if isinstance(ref_scores, np.ndarray):
                ref_scores = ref_scores.tolist()
            
            if pred_scores is None:
                pred_rank_str = ""
            else:
                if len(pred_scores) != len(ref_scores):
                    raise ValueError(
                        f"pred_scores and ref_scores must have the same length, but got {len(pred_scores)} and {len(ref_scores)}"
                    )
                pred_score_dict = {
                    candidate_identifiers[i]: pred_scores[i]
                    for i in range(len(pred_scores))
                }

                pred_rank_str = _score_to_rank(pred_score_dict)

            ref_score_dict = {
                candidate_identifiers[i]: ref_scores[i] for i in range(len(ref_scores))
            }
            ref_rank_str = _score_to_rank(ref_score_dict)
            reward_out = reward_fn(None, pred_rank_str, ref_rank_str)

            if isinstance(reward_out, dict):
                if "score" in reward_out:
                    # report ranking accuracy (ranking_reward)
                    metric_output["ranking_acc"].append(reward_out["score"])
            else:
                metric_output["score"].append(reward_out)

    metric_results = {}
    for metric_name in metric_output.keys():
        metric_results[metric_name] = sum(metric_output[metric_name]) / len(
            metric_output[metric_name]
        )

    return metric_results


def main(
    data_id: tuple[str],
    model_path: str,
    model_name: str,
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    add_prompt_example: bool = False,
    prompt_type: str = "ranking_score",
    runs: int = 4,
    model_type: str = "grrm"
):
    """
    Run ranking accuracy evaluation for reward models on specified datasets.

    This function loads the specified model and datasets, performs inference to generate
    scores/rankings for translation candidates, evaluates the predictions against reference
    scores, and logs the results to Weights & Biases.

    Args:
        data_id: One or more dataset identifiers from RANKING_TEST_DATA_META_INFO.
            Can be a single string (comma-separated), tuple, or iterable of dataset IDs.
        model_path: Path to the pretrained model weights.
        model_name: Name of the model for logging purposes.
        temperature: Sampling temperature for generation. Higher values produce more
            random outputs. Defaults to 0.4.
        top_p: Nucleus sampling probability threshold. Defaults to 0.7.
        max_new_tokens: Maximum number of tokens to generate. Defaults to 4096.
        add_prompt_example: Whether to include examples in the prompt for GRRM, but currently not used.
            Defaults to False.
        prompt_type: Type of prompt template (and model output parser). Only used for GQM models. Must be one of ['score', 'ranking', 
            'ranking_score']. Defaults to "ranking_score".
        runs: Number of inference runs to perform for each sample. Results are
            aggregated across runs. Defaults to 4.
        model_type: Type of model to evaluate. Must be one of ['grrm', 'sqmrm', 'drm'].
            - 'grrm': Group Relative Reward Model (GQM)
            - 'sqmrm': Scalar Quality Metric (SQM) Generative Reward Model
            - 'drm':  Bradley-Terry Reward Model 
            Defaults to "grrm".

    Raises:
        ValueError: If prompt_type is not one of ['score', 'ranking', 'ranking_score'].
        ValueError: If model_type is not one of ['grrm', 'sqmrm', 'drm'].
        ValueError: If data_id is empty or contains invalid dataset identifiers.
        ValueError: If the specified data path does not exist.
    """
    if prompt_type not in ["score", "ranking", "ranking_score"]:
        raise ValueError(
            f"prompt_type must be one of ['score', 'ranking', 'ranking_score']"
        )
    
    if model_type not in ["grrm", "sqmrm", "drm"]:
        raise ValueError(
            f"model_type must be one of ['grrm', 'sqmrm', 'drm']"
        )

    if isinstance(data_id, str):
        data_id_list = tuple(data_id.strip().split(","))
    elif isinstance(data_id, Iterable):
        data_id_list = tuple(data_id)
    else:
        data_id_list = tuple(data_id)
    
    if not data_id_list:
        raise ValueError(f"Invalid data_id. Please provide at least one valid data_id from {RANKING_TEST_DATA_META_INFO.keys()}")

    dfs: Dict[str, pd.DataFrame] = {}
    lang_pairs: Dict[str, str] = {}
    datasets_metric_results = {}
    all_valid_metrics: List[str] = []
    seen_metrics = set()

    for did in data_id_list:
        if did not in RANKING_TEST_DATA_META_INFO:
            raise ValueError(
                f"data_id {did} not in RANKING_TEST_DATA_META_INFO: {RANKING_TEST_DATA_META_INFO.keys()}"
            )
        data_meta_info = RANKING_TEST_DATA_META_INFO[did]
        data_path = Path(data_meta_info["path"])
        lang_pair = f"{data_meta_info['src_lang']}2{data_meta_info['trg_lang']}"

        if not data_path.exists():
            raise ValueError(f"data_path {data_path} does not exist")

        df = pd.read_parquet(data_path)
        dfs[did] = df
        lang_pairs[did] = lang_pair

    if model_type == "grrm":
        from inference.run_rm_GQM import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(model_path)
    elif model_type == "sqmrm":
        from inference.run_rm_SQM import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(model_path)
    elif model_type == "drm":
        from inference.run_drm import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(model_path)

    for did in data_id_list:
        df = dfs[did]
        
        if model_type == "grrm":
            score_for_runs = run_rm_GQM_inference(
                df,
                model_path,
                temperature,
                top_p,
                max_new_tokens,
                prompt_type,
                runs,
                add_prompt_example,
                model=model,
                tokenizer=tokenizer,
            )
        elif model_type == "sqmrm":
            score_for_runs = run_rm_SQM_inference(
                df,
                model_path,
                temperature,
                top_p,
                max_new_tokens,
                runs,
                model=model,
                tokenizer=tokenizer,
            )
        elif model_type == "drm":
            score_for_runs = run_drm_inference(
                df,
                model_path,
                runs,
                model=model,
                tokenizer=tokenizer,
            )

        metric_results = run_eval(df, score_for_runs)
        datasets_metric_results[did] = metric_results
        
        valid_metrics = list(metric_results.keys())
        for m in valid_metrics:
            if m not in seen_metrics:
                seen_metrics.add(m)
                all_valid_metrics.append(m)


    wandb_config = {
        "dataset_names": data_id_list,
        "model_path": model_path,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "prompt_type": prompt_type,
        "runs": runs,
        "metrics": all_valid_metrics,
        "lang_pairs": lang_pairs,
    }

    log_results_to_wandb(
        datasets_metric_results=datasets_metric_results,
        config=wandb_config,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
