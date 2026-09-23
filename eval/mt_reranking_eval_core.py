import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import wandb

from eval.run_mt_eval import (
    _clear_mem,
    _load_datasets,
    _release_vllm_model,
    _sanitize_filename_component,
    _split_scores_by_data_id,
    run_bleurt_eval,
    run_oss_eval,
)
from inference.run_mt import load_model_tokenizer
import inference.run_mt as run_mt
import inference.run_rm_GQM as run_rm_GQM
from utils.config import MT_TEST_DATA_META_INFO
from utils.helpers import (
    build_notes_list,
    load_datasets_from_dir as _load_datasets_from_dir,
)


def log_reranking_results_to_wandb(
    valid_metrics: List[str],
    config: Dict[str, Any],
    datasets_metric_none_counts: Optional[Dict[str, Dict[str, int]]] = None,
    datasets_metric_results: Optional[Dict[str, Dict[str, float]]] = None,
):
    project_name = "mt-reranking-eval"
    wandb.init(project=project_name, name=config["model_name"], config=config)

    columns = ["data_id"] + valid_metrics
    rows: List[List[Any]] = []
    for data_id, metric_dict in (datasets_metric_results or {}).items():
        row: List[Any] = [data_id]
        for m in valid_metrics:
            row.append(metric_dict.get(m, np.nan))
        rows.append(row)

    wandb.log({"metrics_by_dataset": wandb.Table(columns=columns, data=rows)})

    for data_id, metric_dict in (datasets_metric_results or {}).items():
        for m, val in metric_dict.items():
            wandb.run.summary[f"{data_id}/{m}"] = val

    if datasets_metric_none_counts:
        for data_id, metric_none_counts in datasets_metric_none_counts.items():
            for m, cnt in metric_none_counts.items():
                key = f"none_count/{data_id}/{m}"
                try:
                    wandb.run.summary[key] = int(cnt)
                except Exception:
                    wandb.run.summary[key] = cnt


def save_reranking_results_to_json(
    df: pd.DataFrame,
    selected_nested: list[list[str]],
    mt_candidates_first_run: list[list[str]],
    ranking_scores_first_run: list,
    selected_indices_first_run: list[int],
    per_item_metric_avgs: Dict[str, list[Optional[float]]],
    valid_metrics: list[str],
    dataset_name: str,
    model_name: str,
    model_path: str,
    ranking_model_path: str,
    sampling_n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    ranking_temperature: float,
    ranking_top_p: float,
    ranking_max_new_tokens: int,
    runs: int,
    prompt_type: str,
    ranking_prompt_type: str,
    ranking_task_type: str,
    notes_list_per_item: Optional[list[Optional[str]]] = None,
) -> Path:
    safe_model_name = _sanitize_filename_component(model_name)
    safe_dataset_name = _sanitize_filename_component(dataset_name)
    out_file = Path.cwd() / f"{safe_model_name}__reranking__{safe_dataset_name}.json"

    include_notes = notes_list_per_item is not None
    n = len(df)
    items = []
    for i in range(n):
        preds = [selected_nested[r][i] for r in range(runs)]
        metrics_avg_item = {
            m: per_item_metric_avgs.get(m, [None] * n)[i] for m in valid_metrics
        }
        row = df.iloc[i]
        item = {
            "index": int(i),
            "src_lang": str(row["src_lang"]),
            "trg_lang": str(row["trg_lang"]),
            "lang_pair": f"{row['src_lang']}-{row['trg_lang']}",
            "src_text": row["src_text"],
            "ref_text": row["trg_text"],
            "mt_candidates": mt_candidates_first_run[i],
            "selected_translations": preds,
            "ranking_scores": ranking_scores_first_run[i],
            "selected_index": selected_indices_first_run[i],
            "metrics_avg": metrics_avg_item,
        }
        if include_notes:
            item["notes"] = notes_list_per_item[i]
        items.append(item)

    json_payload = {
        "data_name": dataset_name,
        "model_name": model_name,
        "model_path": model_path,
        "ranking_model_path": ranking_model_path,
        "sampling_n": sampling_n,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "ranking_temperature": ranking_temperature,
        "ranking_top_p": ranking_top_p,
        "ranking_max_new_tokens": ranking_max_new_tokens,
        "runs": runs,
        "prompt_type": prompt_type,
        "ranking_prompt_type": ranking_prompt_type,
        "ranking_task_type": ranking_task_type,
        "metrics": valid_metrics,
        "items": items,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    return out_file


def _parse_data_id(data_id) -> tuple[str, ...]:
    if isinstance(data_id, str):
        data_id_list = tuple(data_id.strip().split(","))
    elif isinstance(data_id, Iterable):
        data_id_list = tuple(data_id)
    else:
        data_id_list = (data_id,)

    if not data_id_list:
        raise ValueError(
            f"Invalid data_id. Please provide at least one valid data_id from {MT_TEST_DATA_META_INFO.keys()}"
        )
    return data_id_list


def run_reranking_eval_core(
    data_id: tuple[str],
    model_path: str,
    ranking_model_path: str,
    model_name: str,
    data_dir: Optional[str] = None,
    sampling_n: int = 4,
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    metrics: list[str] = ["bleurt", "oss"],
    prompt_type: str = "codeblock-think",
    ranking_prompt_type: str = "ranking_score",
    ranking_task_type: str = "gqm",
    add_example: bool = False,
    ranking_temperature: Optional[float] = None,
    ranking_top_p: Optional[float] = None,
    ranking_max_new_tokens: Optional[int] = None,
    runs: int = 1,
    save_results: bool = False,
    use_notes: bool = False,
    **kwargs,
):
    if ranking_task_type not in {"gqm", "gqmpe"}:
        raise ValueError("ranking_task_type must be one of {'gqm', 'gqmpe'}")
    if ranking_task_type == "gqmpe" and use_notes:
        raise ValueError("GQMPE reranking does not currently support notes.")
    if ranking_task_type == "gqmpe" and add_example:
        raise ValueError("GQMPE reranking does not support add_example.")

    data_id_list = _parse_data_id(data_id)

    if data_dir:
        df_all, boundaries, dfs_per_id = _load_datasets_from_dir(data_id_list, data_dir)
        lang_pairs = {did: "unknown" for did in data_id_list}
    else:
        df_all, boundaries, lang_pairs, dfs_per_id = _load_datasets(data_id_list)
    N = len(df_all)

    flat_notes_list = None
    if use_notes:
        flat_notes_list = build_notes_list(df_all, runs)
        print(f"Total items: {N}")
    else:
        print(f"Total items: {N}")

    mt_vllm_kwargs = kwargs.get("mt_vllm_kwargs", {})
    model, tokenizer = load_model_tokenizer(model_path, **mt_vllm_kwargs)

    src_list = df_all["src_text"].tolist()
    src_langs = df_all["src_lang"].tolist()
    trg_langs = df_all["trg_lang"].tolist()

    flat_src = src_list * runs
    flat_src_langs = src_langs * runs
    flat_trg_langs = trg_langs * runs

    print(f"Running MT inference: {runs} runs x {N} items x {sampling_n} samples ...")
    mt_output = run_mt.func_call(
        model_path=model_path,
        src_list=flat_src,
        src_langs=flat_src_langs,
        trg_langs=flat_trg_langs,
        sampling_n=sampling_n,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        prompt_type=prompt_type,
        model=model,
        tokenizer=tokenizer,
    )

    mt_responses = mt_output["responses"]
    if sampling_n == 1:
        mt_responses = [[r] for r in mt_responses]

    if ranking_model_path == model_path:
        ranking_model, ranking_tokenizer = model, tokenizer
    else:
        _release_vllm_model(model)
        del model, tokenizer
        ranking_model, ranking_tokenizer = None, None

    if ranking_temperature is None:
        ranking_temperature = temperature
    if ranking_top_p is None:
        ranking_top_p = top_p
    if ranking_max_new_tokens is None:
        ranking_max_new_tokens = max_new_tokens

    print(f"Running {ranking_task_type.upper()} ranking: {runs * N} items ...")
    if ranking_model is None:
        ranking_vllm_kwargs = kwargs.get("ranking_vllm_kwargs", {})
        ranking_model, ranking_tokenizer = load_model_tokenizer(
            ranking_model_path, **ranking_vllm_kwargs
        )

    ranking_kwargs = dict(
        model_path=ranking_model_path,
        src_list=flat_src,
        mt_list=mt_responses,
        src_langs=flat_src_langs,
        trg_langs=flat_trg_langs,
        temperature=ranking_temperature,
        top_p=ranking_top_p,
        max_new_tokens=ranking_max_new_tokens,
        prompt_type=ranking_prompt_type,
        task_type=ranking_task_type,
        add_example=add_example,
        model=ranking_model,
        tokenizer=ranking_tokenizer,
    )
    if use_notes:
        ranking_kwargs["notes_list"] = flat_notes_list
    ranking_output = run_rm_GQM.func_call(**ranking_kwargs)

    ranking_scores = ranking_output["scores"]

    selected_flat = []
    selected_indices_flat = []
    for scores, candidates in zip(ranking_scores, mt_responses):
        if scores is not None:
            best_idx = max(range(len(scores)), key=lambda j: scores[j])
            selected_flat.append(candidates[best_idx])
            selected_indices_flat.append(best_idx)
        else:
            selected_flat.append(candidates[0])
            selected_indices_flat.append(0)

    try:
        _release_vllm_model(ranking_model)
        del ranking_model, ranking_tokenizer
        if "model" in dir() and model is not None:
            del model, tokenizer
    except Exception:
        pass
    _clear_mem()

    bleurt_model_path = kwargs.get("bleurt_model_path")
    oss_model_path = kwargs.get("oss_model_path")

    datasets_metric_results: Dict[str, Dict[str, float]] = {did: {} for did in data_id_list}
    datasets_metric_none_counts: Dict[str, Dict[str, int]] = {did: {} for did in data_id_list}
    datasets_per_item_metric_avgs: Dict[str, Dict[str, list[Optional[float]]]] = {
        did: {} for did in data_id_list
    }
    datasets_valid_metrics: Dict[str, List[str]] = {did: [] for did in data_id_list}

    all_valid_metrics: List[str] = []
    seen_metrics = set()

    def add_metric(metric_name: str, scores_flat: list[float]):
        metric_results = _split_scores_by_data_id(scores_flat, boundaries, N, runs)
        for did in data_id_list:
            datasets_metric_results[did][metric_name] = metric_results[did]["avg"]
            datasets_metric_none_counts[did][metric_name] = metric_results[did]["none_count"]
            datasets_per_item_metric_avgs[did][metric_name] = metric_results[did]["per_item_avgs"]
            datasets_valid_metrics[did].append(metric_name)

        if metric_name not in seen_metrics:
            seen_metrics.add(metric_name)
            all_valid_metrics.append(metric_name)

    if "oss" in metrics:
        if oss_model_path is None:
            oss_model_path = "openai/gpt-oss-120b"
        import inference.run_oss_SQM as run_oss_SQM

        oss_vllm_kwargs = kwargs.get("oss_vllm_kwargs", {})
        oss_model = run_oss_SQM.init_oss_model(oss_model_path, **oss_vllm_kwargs)
        oss_scores_flat = run_oss_eval(
            df_all, selected_flat, runs, oss_model, oss_model_path=oss_model_path
        )
        add_metric("oss", oss_scores_flat)

        try:
            _release_vllm_model(oss_model)
            del oss_model
        except Exception:
            pass

    if "bleurt" in metrics:
        bleurt_scores_flat = run_bleurt_eval(
            df_all, selected_flat, runs, bleurt_model_path=bleurt_model_path
        )
        add_metric("bleurt", bleurt_scores_flat)

    for did in data_id_list:
        print(f"\n=== {did} ===")
        for m in datasets_valid_metrics[did]:
            avg = datasets_metric_results[did][m]
            none_count = datasets_metric_none_counts[did][m]
            print(f"  {m}: {avg:.4f} (none_count={none_count})")

    if save_results:
        for did, (start, end) in boundaries.items():
            n = end - start
            selected_nested = [
                [selected_flat[r * N + start + i] for i in range(n)]
                for r in range(runs)
            ]
            mt_candidates_first_run = [mt_responses[start + i] for i in range(n)]
            ranking_scores_first_run = [ranking_scores[start + i] for i in range(n)]
            selected_indices_first_run = [selected_indices_flat[start + i] for i in range(n)]

            save_kwargs = {}
            if use_notes:
                save_kwargs = {
                    "notes_list_per_item": flat_notes_list[start:end],
                }

            save_reranking_results_to_json(
                df=dfs_per_id[did],
                selected_nested=selected_nested,
                mt_candidates_first_run=mt_candidates_first_run,
                ranking_scores_first_run=ranking_scores_first_run,
                selected_indices_first_run=selected_indices_first_run,
                per_item_metric_avgs=datasets_per_item_metric_avgs[did],
                valid_metrics=datasets_valid_metrics[did],
                dataset_name=did,
                model_name=model_name,
                model_path=model_path,
                ranking_model_path=ranking_model_path,
                sampling_n=sampling_n,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                ranking_temperature=ranking_temperature,
                ranking_top_p=ranking_top_p,
                ranking_max_new_tokens=ranking_max_new_tokens,
                runs=runs,
                prompt_type=prompt_type,
                ranking_prompt_type=ranking_prompt_type,
                ranking_task_type=ranking_task_type,
                **save_kwargs,
            )

    wandb_config = {
        "dataset_names": data_id_list,
        "model_path": model_path,
        "ranking_model_path": ranking_model_path,
        "model_name": model_name,
        "temperature": temperature,
        "ranking_temperature": ranking_temperature,
        "top_p": top_p,
        "ranking_top_p": ranking_top_p,
        "max_new_tokens": max_new_tokens,
        "ranking_max_new_tokens": ranking_max_new_tokens,
        "runs": runs,
        "sampling_n": sampling_n,
        "metrics": all_valid_metrics,
        "lang_pairs": lang_pairs,
        "prompt_type": prompt_type,
        "ranking_prompt_type": ranking_prompt_type,
        "ranking_task_type": ranking_task_type,
        "add_example": add_example,
        "data_dir": data_dir,
    }
    log_reranking_results_to_wandb(
        valid_metrics=all_valid_metrics,
        config=wandb_config,
        datasets_metric_none_counts=datasets_metric_none_counts,
        datasets_metric_results=datasets_metric_results,
    )
