import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import wandb

from eval.run_mt_eval import (
    _clear_mem,
    _load_datasets,
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
    split_metrics_by_notes,
)


def log_reranking_results_to_wandb(
    valid_metrics: List[str],
    config: Dict[str, Any],
    datasets_metric_none_counts: Optional[Dict[str, Dict[str, int]]] = None,
    datasets_metric_results: Optional[Dict[str, Dict[str, float]]] = None,
    datasets_metrics_by_notes: Optional[Dict[str, Dict[str, dict]]] = None,
):
    project_name = "mt-reranking-eval"
    wandb.init(project=project_name, name=config["model_name"], config=config)

    if datasets_metrics_by_notes is not None:
        columns = ["data_id"]
        for m in valid_metrics:
            columns.extend([f"{m}/all", f"{m}/notes", f"{m}/no_notes"])
        columns.extend(["notes_count", "no_notes_count"])

        rows: List[List[Any]] = []
        for data_id, metric_dict in datasets_metrics_by_notes.items():
            row: List[Any] = [data_id]
            for m in valid_metrics:
                split = metric_dict.get(m, {})
                row.append(split.get("all", {}).get("avg", np.nan))
                row.append(split.get("notes", {}).get("avg", np.nan))
                row.append(split.get("no_notes", {}).get("avg", np.nan))
            first_split = next(iter(metric_dict.values()), {})
            row.append(first_split.get("notes", {}).get("count", 0))
            row.append(first_split.get("no_notes", {}).get("count", 0))
            rows.append(row)

        wandb.log({"metrics_by_dataset": wandb.Table(columns=columns, data=rows)})

        for data_id, metric_dict in datasets_metrics_by_notes.items():
            for m, split in metric_dict.items():
                for group in ("all", "notes", "no_notes"):
                    wandb.run.summary[f"{data_id}/{m}/{group}"] = split.get(group, {}).get("avg", np.nan)
            first_split = next(iter(metric_dict.values()), {})
            wandb.run.summary[f"{data_id}/notes_count"] = first_split.get("notes", {}).get("count", 0)
            wandb.run.summary[f"{data_id}/no_notes_count"] = first_split.get("no_notes", {}).get("count", 0)
    else:
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
    notes_list_per_item: Optional[list[Optional[str]]] = None,
    use_notes_mask_per_item: Optional[list[bool]] = None,
    difficulty_list_per_item: Optional[list] = None,
    difficulty_filter: Optional[int] = None,
) -> Path:
    safe_model_name = _sanitize_filename_component(model_name)
    safe_dataset_name = _sanitize_filename_component(dataset_name)
    out_file = Path.cwd() / f"{safe_model_name}__reranking__{safe_dataset_name}.json"

    include_notes = notes_list_per_item is not None and use_notes_mask_per_item is not None
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
            item["use_notes"] = use_notes_mask_per_item[i]
            item["difficulty"] = difficulty_list_per_item[i] if difficulty_list_per_item is not None else 0
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
        "metrics": valid_metrics,
        "items": items,
    }
    if include_notes:
        json_payload["difficulty_filter"] = difficulty_filter

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


def _difficulty_list(df: pd.DataFrame) -> list:
    if "difficulty" not in df.columns:
        return [0] * len(df)
    return df["difficulty"].tolist()


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
    add_example: bool = False,
    ranking_temperature: Optional[float] = None,
    ranking_top_p: Optional[float] = None,
    ranking_max_new_tokens: Optional[int] = None,
    runs: int = 1,
    save_results: bool = False,
    use_notes: bool = False,
    difficulty_filter: int = 0,
    **kwargs,
):
    data_id_list = _parse_data_id(data_id)

    if data_dir:
        df_all, boundaries, dfs_per_id = _load_datasets_from_dir(data_id_list, data_dir)
        lang_pairs = {did: "unknown" for did in data_id_list}
    else:
        df_all, boundaries, lang_pairs, dfs_per_id = _load_datasets(data_id_list)
    N = len(df_all)

    flat_notes_list = None
    flat_use_notes_mask = None
    if use_notes:
        flat_notes_list, flat_use_notes_mask = build_notes_list(df_all, difficulty_filter, runs)
        notes_used = sum(flat_use_notes_mask[:N])
        print(f"Total items: {N}, notes used: {notes_used}/{N} (difficulty_filter={difficulty_filter})")
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
        del model, tokenizer
        _clear_mem()
        ranking_model, ranking_tokenizer = None, None

    if ranking_temperature is None:
        ranking_temperature = temperature
    if ranking_top_p is None:
        ranking_top_p = top_p
    if ranking_max_new_tokens is None:
        ranking_max_new_tokens = max_new_tokens

    print(f"Running GQM ranking: {runs * N} items ...")
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
        del ranking_model, ranking_tokenizer
        if "model" in dir() and model is not None:
            del model, tokenizer
    except Exception:
        pass
    _clear_mem()

    bleurt_model_path = kwargs.get("bleurt_model_path")
    oss_model_path = kwargs.get("oss_model_path")

    datasets_metric_results: Dict[str, Dict[str, float]] = {did: {} for did in data_id_list}
    datasets_metrics_by_notes: Optional[Dict[str, Dict[str, dict]]] = (
        {did: {} for did in data_id_list} if use_notes else None
    )
    datasets_metric_none_counts: Dict[str, Dict[str, int]] = {did: {} for did in data_id_list}
    datasets_per_item_metric_avgs: Dict[str, Dict[str, list[Optional[float]]]] = {
        did: {} for did in data_id_list
    }
    datasets_valid_metrics: Dict[str, List[str]] = {did: [] for did in data_id_list}

    all_valid_metrics: List[str] = []
    seen_metrics = set()

    def add_metric(metric_name: str, scores_flat: list[float]):
        if use_notes:
            metric_split = split_metrics_by_notes(scores_flat, flat_use_notes_mask, boundaries, N, runs)
            for did in data_id_list:
                datasets_metrics_by_notes[did][metric_name] = metric_split[did]
                datasets_metric_none_counts[did][metric_name] = metric_split[did]["all"]["none_count"]
                datasets_per_item_metric_avgs[did][metric_name] = metric_split[did]["all"]["per_item_avgs"]
                datasets_valid_metrics[did].append(metric_name)
        else:
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
            del oss_model
            _clear_mem()
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
            if use_notes:
                split = datasets_metrics_by_notes[did][m]
                all_avg = split["all"]["avg"]
                notes_avg = split["notes"]["avg"]
                no_notes_avg = split["no_notes"]["avg"]
                notes_cnt = split["notes"]["count"]
                no_notes_cnt = split["no_notes"]["count"]
                print(f"  {m}: all={all_avg:.4f} | notes({notes_cnt})={notes_avg:.4f} | no_notes({no_notes_cnt})={no_notes_avg:.4f}")
            else:
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
                    "use_notes_mask_per_item": flat_use_notes_mask[start:end],
                    "difficulty_list_per_item": _difficulty_list(df_all.iloc[start:end]),
                    "difficulty_filter": difficulty_filter,
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
        "add_example": add_example,
        "data_dir": data_dir,
    }
    if use_notes:
        wandb_config["difficulty_filter"] = difficulty_filter

    log_reranking_results_to_wandb(
        valid_metrics=all_valid_metrics,
        config=wandb_config,
        datasets_metric_none_counts=datasets_metric_none_counts,
        datasets_metric_results=None if use_notes else datasets_metric_results,
        datasets_metrics_by_notes=datasets_metrics_by_notes if use_notes else None,
    )
