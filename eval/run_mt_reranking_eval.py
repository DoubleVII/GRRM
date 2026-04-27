import pandas as pd
import numpy as np
from pathlib import Path
import json
import wandb

from typing import Optional, Dict, Any, List, Iterable

from eval.run_mt_eval import (
    _load_datasets,
    _split_scores_by_data_id,
    run_bleurt_eval,
    run_oss_eval,
    _normalize_metric_output,
    _average_overall,
    _average_per_item,
    _clear_mem,
    _sanitize_filename_component,
)
from eval.run_mt_gpe_eval import (
    _load_datasets_from_dir,
    _build_notes_list,
    _split_metrics_by_notes,
)
from inference.run_mt import load_model_tokenizer
import inference.run_mt as run_mt
import inference.run_rm_GQM as run_rm_GQM
from utils.config import MT_TEST_DATA_META_INFO


def log_reranking_results_to_wandb(
    datasets_metrics_by_notes: Dict[str, Dict[str, dict]],
    valid_metrics: List[str],
    config: Dict[str, Any],
    datasets_metric_none_counts: Optional[Dict[str, Dict[str, int]]] = None,
):
    project_name = "mt-reranking-eval"
    wandb.init(project=project_name, name=config["model_name"], config=config)

    columns = ["data_id"]
    for m in valid_metrics:
        columns.extend([f"{m}/all", f"{m}/notes", f"{m}/no_notes"])
    columns.extend(["notes_count", "no_notes_count"])

    rows: List[List[Any]] = []
    for data_id, metric_dict in datasets_metrics_by_notes.items():
        row: List[Any] = [data_id]
        for m in valid_metrics:
            split = metric_dict.get(m, {})
            all_data = split.get("all", {})
            notes_data = split.get("notes", {})
            no_notes_data = split.get("no_notes", {})
            row.append(all_data.get("avg", np.nan))
            row.append(notes_data.get("avg", np.nan))
            row.append(no_notes_data.get("avg", np.nan))
        first_split = next(iter(metric_dict.values()), {})
        notes_count = first_split.get("notes", {}).get("count", 0)
        no_notes_count = first_split.get("no_notes", {}).get("count", 0)
        row.append(notes_count)
        row.append(no_notes_count)
        rows.append(row)

    table = wandb.Table(columns=columns, data=rows)
    wandb.log({"metrics_by_dataset": table})

    for data_id, metric_dict in datasets_metrics_by_notes.items():
        for m, split in metric_dict.items():
            for group in ("all", "notes", "no_notes"):
                key = f"{data_id}/{m}/{group}"
                val = split.get(group, {}).get("avg", np.nan)
                wandb.run.summary[key] = val
        first_split = next(iter(metric_dict.values()), {})
        wandb.run.summary[f"{data_id}/notes_count"] = first_split.get("notes", {}).get("count", 0)
        wandb.run.summary[f"{data_id}/no_notes_count"] = first_split.get("no_notes", {}).get("count", 0)

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
    notes_list_per_item: list[Optional[str]],
    use_notes_mask_per_item: list[bool],
    difficulty_list_per_item: list,
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
    difficulty_filter: int,
) -> Path:
    safe_model_name = _sanitize_filename_component(model_name)
    safe_dataset_name = _sanitize_filename_component(dataset_name)
    out_file = Path.cwd() / f"{safe_model_name}__reranking__{safe_dataset_name}.json"

    n = len(df)
    items = []
    for i in range(n):
        preds = [selected_nested[r][i] for r in range(runs)]
        metrics_avg_item = {
            m: per_item_metric_avgs.get(m, [None] * n)[i] for m in valid_metrics
        }
        row = df.iloc[i]
        items.append({
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
            "notes": notes_list_per_item[i],
            "use_notes": use_notes_mask_per_item[i],
            "difficulty": difficulty_list_per_item[i],
            "metrics_avg": metrics_avg_item,
        })

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
        "difficulty_filter": difficulty_filter,
        "metrics": valid_metrics,
        "items": items,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    return out_file


def main(
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
    difficulty_filter: int = 0,
    **kwargs,
):
    # --- Parse data_id ---
    if isinstance(data_id, str):
        data_id_list = tuple(data_id.strip().split(","))
    elif isinstance(data_id, Iterable):
        data_id_list = tuple(data_id)
    else:
        data_id_list = (data_id,)

    if not data_id_list:
        raise ValueError("Please provide at least one data_id")

    # --- Load datasets ---
    if data_dir:
        df_all, boundaries, dfs_per_id = _load_datasets_from_dir(
            data_id_list, data_dir
        )
        lang_pairs = {did: "unknown" for did in data_id_list}
    else:
        df_all, boundaries, lang_pairs, dfs_per_id = _load_datasets(data_id_list)
    N = len(df_all)

    # --- Build notes list ---
    flat_notes_list, flat_use_notes_mask = _build_notes_list(
        df_all, difficulty_filter, runs
    )
    notes_used = sum(flat_use_notes_mask[:N])
    print(f"Total items: {N}, notes used: {notes_used}/{N} (difficulty_filter={difficulty_filter})")

    # --- Load MT model ---
    mt_vllm_kwargs = kwargs.get("mt_vllm_kwargs", {})
    model, tokenizer = load_model_tokenizer(model_path, **mt_vllm_kwargs)

    # --- Build flat inputs (run-major order) ---
    src_list = df_all["src_text"].tolist()
    src_langs = df_all["src_lang"].tolist()
    trg_langs = df_all["trg_lang"].tolist()

    flat_src = src_list * runs
    flat_src_langs = src_langs * runs
    flat_trg_langs = trg_langs * runs

    # --- Stage 1: MT inference with sampling_n ---
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

    # --- Conditionally release MT model ---
    if ranking_model_path == model_path:
        ranking_model, ranking_tokenizer = model, tokenizer
    else:
        del model, tokenizer
        _clear_mem()
        ranking_model, ranking_tokenizer = None, None

    # --- Stage 2: GQM ranking + best-candidate selection ---
    if ranking_temperature is None:
        ranking_temperature = temperature
    if ranking_top_p is None:
        ranking_top_p = top_p
    if ranking_max_new_tokens is None:
        ranking_max_new_tokens = max_new_tokens

    print(f"Running GQM ranking: {runs * N} items ...")
    if ranking_model is None:
        ranking_vllm_kwargs = kwargs.get("ranking_vllm_kwargs", {})
        ranking_model, ranking_tokenizer = load_model_tokenizer(ranking_model_path, **ranking_vllm_kwargs)

    ranking_output = run_rm_GQM.func_call(
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
        notes_list=flat_notes_list,
    )

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

    # Release ranking model
    try:
        del ranking_model, ranking_tokenizer
        if "model" in dir() and model is not None:
            del model, tokenizer
    except Exception:
        pass
    _clear_mem()

    # --- Stage 3: Evaluation ---
    bleurt_model_path = kwargs.get("bleurt_model_path")
    oss_model_path = kwargs.get("oss_model_path")

    datasets_metrics_by_notes: Dict[str, Dict[str, dict]] = {
        did: {} for did in data_id_list
    }
    datasets_metric_none_counts: Dict[str, Dict[str, int]] = {
        did: {} for did in data_id_list
    }
    datasets_per_item_metric_avgs: Dict[str, Dict[str, list[Optional[float]]]] = {
        did: {} for did in data_id_list
    }
    datasets_valid_metrics: Dict[str, List[str]] = {did: [] for did in data_id_list}

    all_valid_metrics: List[str] = []
    seen_metrics = set()

    if "oss" in metrics:
        if oss_model_path is None:
            oss_model_path = "openai/gpt-oss-120b"
        import inference.run_oss_SQM as run_oss_SQM

        oss_vllm_kwargs = kwargs.get("oss_vllm_kwargs", {})
        oss_model = run_oss_SQM.init_oss_model(oss_model_path, **oss_vllm_kwargs)

        oss_scores_flat = run_oss_eval(
            df_all, selected_flat, runs, oss_model, oss_model_path=oss_model_path
        )
        oss_notes_split = _split_metrics_by_notes(
            oss_scores_flat, flat_use_notes_mask, boundaries, N, runs
        )

        for did in data_id_list:
            datasets_metrics_by_notes[did]["oss"] = oss_notes_split[did]
            datasets_metric_none_counts[did]["oss"] = oss_notes_split[did]["all"]["none_count"]
            datasets_per_item_metric_avgs[did]["oss"] = oss_notes_split[did]["all"]["per_item_avgs"]
            datasets_valid_metrics[did].append("oss")

        if "oss" not in seen_metrics:
            seen_metrics.add("oss")
            all_valid_metrics.append("oss")

        try:
            del oss_model
            _clear_mem()
        except Exception:
            pass

    if "bleurt" in metrics:
        bleurt_scores_flat = run_bleurt_eval(
            df_all, selected_flat, runs, bleurt_model_path=bleurt_model_path
        )
        bleurt_notes_split = _split_metrics_by_notes(
            bleurt_scores_flat, flat_use_notes_mask, boundaries, N, runs
        )

        for did in data_id_list:
            datasets_metrics_by_notes[did]["bleurt"] = bleurt_notes_split[did]
            datasets_metric_none_counts[did]["bleurt"] = bleurt_notes_split[did]["all"]["none_count"]
            datasets_per_item_metric_avgs[did]["bleurt"] = bleurt_notes_split[did]["all"]["per_item_avgs"]
            datasets_valid_metrics[did].append("bleurt")

        if "bleurt" not in seen_metrics:
            seen_metrics.add("bleurt")
            all_valid_metrics.append("bleurt")

    # --- Print summary ---
    for did in data_id_list:
        print(f"\n=== {did} ===")
        for m in datasets_valid_metrics[did]:
            split = datasets_metrics_by_notes[did][m]
            all_avg = split["all"]["avg"]
            notes_avg = split["notes"]["avg"]
            no_notes_avg = split["no_notes"]["avg"]
            notes_cnt = split["notes"]["count"]
            no_notes_cnt = split["no_notes"]["count"]
            print(f"  {m}: all={all_avg:.4f} | notes({notes_cnt})={notes_avg:.4f} | no_notes({no_notes_cnt})={no_notes_avg:.4f}")

    # --- Save results ---
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
            notes_per_item = flat_notes_list[start:end]
            mask_per_item = flat_use_notes_mask[start:end]
            difficulty_per_item = df_all.iloc[start:end].get("difficulty", [0] * n).tolist()

            save_reranking_results_to_json(
                df=dfs_per_id[did],
                selected_nested=selected_nested,
                mt_candidates_first_run=mt_candidates_first_run,
                ranking_scores_first_run=ranking_scores_first_run,
                selected_indices_first_run=selected_indices_first_run,
                per_item_metric_avgs=datasets_per_item_metric_avgs[did],
                valid_metrics=datasets_valid_metrics[did],
                notes_list_per_item=notes_per_item,
                use_notes_mask_per_item=mask_per_item,
                difficulty_list_per_item=difficulty_per_item,
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
                difficulty_filter=difficulty_filter,
            )

    # --- Wandb logging ---
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
        "difficulty_filter": difficulty_filter,
        "data_dir": data_dir,
    }

    log_reranking_results_to_wandb(
        datasets_metrics_by_notes=datasets_metrics_by_notes,
        valid_metrics=all_valid_metrics,
        config=wandb_config,
        datasets_metric_none_counts=datasets_metric_none_counts,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
