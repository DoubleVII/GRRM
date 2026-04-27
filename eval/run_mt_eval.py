import pandas as pd
from pathlib import Path
import json

from typing import Optional, Dict, Any, List, Iterable
import wandb

from utils.config import MT_TEST_DATA_META_INFO
import numpy as np
from utils.helpers import (
    build_notes_list,
    split_metrics_by_notes,
    average_overall,
    average_per_item,
    load_datasets_from_dir,
)
from inference.run_mt import load_model_tokenizer
import inference.run_oss_SQM as run_oss_SQM

# Backward-compatible aliases for external importers
_average_overall = average_overall
_average_per_item = average_per_item

def log_results_to_wandb(
    datasets_metric_results: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    datasets_metric_none_counts: Optional[Dict[str, Dict[str, int]]] = None,
    datasets_metrics_by_notes: Optional[Dict[str, Dict[str, dict]]] = None,
):
    """Log results from multiple datasets to wandb as a table.

    Args:
        datasets_metric_results: {dataset_name: {metric: value}}
        config: wandb config
        datasets_metric_none_counts: {dataset_name: {metric: none_count}}
        datasets_metrics_by_notes: {data_id: {metric: {"all", "notes", "no_notes"}}}
            When provided, also logs a notes-breakdown table.
    """

    project_name = "mt-eval"
    wandb.init(
        project=project_name,
        name=config["model_name"],
        config=config,
    )

    # All metrics that have appeared
    all_metrics: List[str] = []
    seen = set()
    for _ds, mr in datasets_metric_results.items():
        for m in mr.keys():
            if m not in seen:
                seen.add(m)
                all_metrics.append(m)

    # Use config["metrics"] as primary, fallback to all metrics if empty
    cfg_metrics = config.get("metrics") or []
    if cfg_metrics:
        metrics = [m for m in cfg_metrics if m in seen] or all_metrics
    else:
        metrics = all_metrics

    # Build table aggregated by dataset, first column is data_id (i.e., dataset_name)
    columns = ["data_id"] + metrics
    rows: List[List[Any]] = []
    for dataset_name, metric_results in datasets_metric_results.items():
        row = [dataset_name]
        for m in metrics:
            row.append(metric_results.get(m, np.nan))
        rows.append(row)

    table = wandb.Table(columns=columns, data=rows)

    # Log to wandb
    wandb.log({"metrics_by_dataset": table})

    # Sync to summary for quick access (data_id/metric)
    for dataset_name, metric_results in datasets_metric_results.items():
        for m, v in metric_results.items():
            wandb.run.summary[f"{dataset_name}/{m}"] = v

    # Log None counts to summary
    if datasets_metric_none_counts:
        for dataset_name, metric_none_counts in datasets_metric_none_counts.items():
            for m, cnt in metric_none_counts.items():
                key = f"none_count/{dataset_name}/{m}"
                try:
                    wandb.run.summary[key] = int(cnt)
                except Exception:
                    wandb.run.summary[key] = cnt

    # Log notes-breakdown table and summary
    if datasets_metrics_by_notes:
        cfg_metrics = config.get("metrics") or []
        notes_metrics = [m for m in cfg_metrics if m in datasets_metrics_by_notes.get(
            next(iter(datasets_metrics_by_notes)), {}
        )] if datasets_metrics_by_notes else []

        if not notes_metrics:
            # Derive from the data
            seen_m: set[str] = set()
            notes_metrics = []
            for did, md in datasets_metrics_by_notes.items():
                for m in md:
                    if m not in seen_m:
                        seen_m.add(m)
                        notes_metrics.append(m)

        columns_nb = ["data_id"]
        for m in notes_metrics:
            columns_nb.extend([f"{m}/all", f"{m}/notes", f"{m}/no_notes"])
        columns_nb.extend(["notes_count", "no_notes_count"])

        rows_nb: List[List[Any]] = []
        for data_id, metric_dict in datasets_metrics_by_notes.items():
            row: List[Any] = [data_id]
            for m in notes_metrics:
                split = metric_dict.get(m, {})
                row.append(split.get("all", {}).get("avg", np.nan))
                row.append(split.get("notes", {}).get("avg", np.nan))
                row.append(split.get("no_notes", {}).get("avg", np.nan))
            first_split = next(iter(metric_dict.values()), {})
            row.append(first_split.get("notes", {}).get("count", 0))
            row.append(first_split.get("no_notes", {}).get("count", 0))
            rows_nb.append(row)

        table_nb = wandb.Table(columns=columns_nb, data=rows_nb)
        wandb.log({"metrics_by_notes": table_nb})

        for data_id, metric_dict in datasets_metrics_by_notes.items():
            for m, split in metric_dict.items():
                for group in ("all", "notes", "no_notes"):
                    key = f"{data_id}/{m}/{group}"
                    val = split.get(group, {}).get("avg", np.nan)
                    wandb.run.summary[key] = val
            first_split = next(iter(metric_dict.values()), {})
            wandb.run.summary[f"{data_id}/notes_count"] = first_split.get("notes", {}).get("count", 0)
            wandb.run.summary[f"{data_id}/no_notes_count"] = first_split.get("no_notes", {}).get("count", 0)


def _sanitize_filename_component(s: str) -> str:
    try:
        import re as _re
    except Exception:
        _re = None
    s = str(s).strip()
    if _re is not None:
        s = _re.sub(r"[\\/]+", "_", s)
    return s.replace(" ", "_")


def save_results_to_json(
    df: pd.DataFrame,
    mt_list_for_runs_nested: list[list[str]],
    per_item_metric_avgs: Dict[str, list[Optional[float]]],
    valid_metrics: list[str],
    dataset_name: str,
    model_name: str,
    model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    runs: int,
    prompt_type: str,
    difficulty_filter: int = 0,
    notes_list_per_item: Optional[list[Optional[str]]] = None,
    use_notes_mask_per_item: Optional[list[bool]] = None,
) -> Path:
    safe_model_name = _sanitize_filename_component(model_name)
    safe_dataset_name = _sanitize_filename_component(dataset_name)
    out_file = Path.cwd() / f"{safe_model_name}__{safe_dataset_name}.json"

    n = len(df)
    items = []
    for i in range(n):
        preds = [mt_list_for_runs_nested[r][i] for r in range(runs)]
        metrics_avg_item = {m: per_item_metric_avgs.get(m, [None] * n)[i] for m in valid_metrics}
        row = df.iloc[i]
        items.append({
            "index": int(i),
            "src_lang": str(row["src_lang"]),
            "trg_lang": str(row["trg_lang"]),
            "lang_pair": f"{row['src_lang']}-{row['trg_lang']}",
            "src_text": row["src_text"],
            "ref_text": row["trg_text"],
            "predictions": preds,
            "metrics_avg": metrics_avg_item,
            "notes": notes_list_per_item[i] if notes_list_per_item else None,
            "use_notes": use_notes_mask_per_item[i] if use_notes_mask_per_item else None,
        })

    json_payload = {
        "data_name": dataset_name,
        "model_name": model_name,
        "model_path": model_path,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "runs": runs,
        "prompt_type": prompt_type,
        "difficulty_filter": difficulty_filter,
        "metrics": valid_metrics,
        "items": items,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    return out_file



def _load_datasets(
    data_id_list: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]], dict[str, str], dict[str, pd.DataFrame]]:
    """Load and concatenate all parquets for the given data_ids.

    Returns:
        df_all: concatenated DataFrame with a ``_data_id`` column.
        boundaries: ``{data_id: (start_idx, end_idx)}`` into df_all rows.
        lang_pairs: ``{data_id: "src2trg"}`` from config.
        dfs_per_id: ``{data_id: original DataFrame}`` for save_results.
    """
    frames: list[pd.DataFrame] = []
    boundaries: dict[str, tuple[int, int]] = {}
    lang_pairs: dict[str, str] = {}
    dfs_per_id: dict[str, pd.DataFrame] = {}

    offset = 0
    for did in data_id_list:
        if did not in MT_TEST_DATA_META_INFO:
            raise ValueError(
                f"data_id {did} not in MT_TEST_DATA_META_INFO: {MT_TEST_DATA_META_INFO.keys()}"
            )
        meta = MT_TEST_DATA_META_INFO[did]
        data_path = Path(meta["path"])
        if not data_path.exists():
            raise ValueError(f"data_path {data_path} does not exist")

        df = pd.read_parquet(data_path)
        df["_data_id"] = did
        n = len(df)
        boundaries[did] = (offset, offset + n)
        lang_pairs[did] = f"{meta['src_lang']}2{meta['trg_lang']}"
        dfs_per_id[did] = df

        frames.append(df)
        offset += n

    df_all = pd.concat(frames, ignore_index=True)
    return df_all, boundaries, lang_pairs, dfs_per_id


def _split_scores_by_data_id(
    scores_flat: list[float],
    boundaries: dict[str, tuple[int, int]],
    total_n: int,
    runs: int,
) -> dict[str, dict]:
    """Split flat run-major scores back per data_id and compute averages.

    Args:
        scores_flat: length ``runs * total_n``, run-major order.
        boundaries: ``{data_id: (start, end)}`` into the total_n items.
        total_n: total number of items across all data_ids.
        runs: number of inference runs.

    Returns:
        ``{data_id: {"avg": float, "none_count": int, "per_item_avgs": list}}``
    """
    results: dict[str, dict] = {}
    for did, (start, end) in boundaries.items():
        n = end - start
        # Collect scores for this data_id across all runs (run-major order)
        did_scores: list[float] = []
        for r in range(runs):
            did_scores.extend(scores_flat[r * total_n + start : r * total_n + end])

        avg, none_count = _average_overall(did_scores)
        per_item_avgs = _average_per_item(did_scores, n, runs)
        results[did] = {"avg": avg, "none_count": none_count, "per_item_avgs": per_item_avgs}
    return results


def run_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    prompt_type: str,
    runs: int,
    notes_list: Optional[list[Optional[str]]] = None,
) -> list[str]:
    """Run MT inference on the concatenated dataset, single batched call.

    Returns:
        Flat list of length ``runs * len(df)`` in run-major order.
    """
    import inference.run_mt as run_mt

    src_list = df["src_text"].tolist()
    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()

    flat_src = src_list * runs
    flat_src_langs = src_langs * runs
    flat_trg_langs = trg_langs * runs

    func_call_kwargs = {
        "model_path": model_path,
        "src_list": flat_src,
        "src_langs": flat_src_langs,
        "trg_langs": flat_trg_langs,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "prompt_type": prompt_type,
        "model": model,
        "tokenizer": tokenizer,
        "notes_list": notes_list,
    }
    if "Seed-X" in model_path:
        func_call_kwargs["use_chat_template"] = False
    output_dict = run_mt.func_call(**func_call_kwargs)
    mt_flat = output_dict["responses"]

    expected_len = runs * len(df)
    if len(mt_flat) != expected_len:
        raise ValueError(
            f"mt_flat must have length {expected_len} (runs={runs} * items={len(df)}), "
            f"but got {len(mt_flat)}"
        )

    return mt_flat


def _normalize_metric_output(output: Any, n_items: int, n_runs: int) -> list[float]:
    try:
        import numpy as _np
    except Exception:
        _np = None

    if isinstance(output, dict):
        if "scores" in output:
            output = output["scores"]
        elif "bleurt_scores" in output:
            output = output["bleurt_scores"]

    if _np is not None and isinstance(output, _np.ndarray):
        output = output.tolist()

    if isinstance(output, list):
        if len(output) == n_runs and len(output) > 0 and isinstance(output[0], list):
            return [v for sub in output for v in sub]
        if len(output) == n_items * n_runs:
            return output

    raise ValueError(f"Unexpected metric output shape/type: type={type(output)}, len={getattr(output, '__len__', 'NA')}")

def run_bleurt_eval(
    df: pd.DataFrame,
    mt_flat: list[str],
    runs: int,
    bleurt_model_path: Optional[str] = None,
) -> list[float]:
    """Run BLEURT evaluation on the concatenated dataset, single batched call.

    Args:
        df: concatenated DataFrame.
        mt_flat: flat predictions, length ``runs * len(df)``, run-major order.
        runs: number of inference runs.
        bleurt_model_path: path to the BLEURT model checkpoint.

    Returns:
        Flat BLEURT scores, length ``runs * len(df)``, run-major order.
    """
    N = len(df)
    if N == 0:
        raise ValueError("Input data is empty: df has 0 rows")

    ref_list = df["trg_text"].tolist()
    flat_ref = ref_list * runs

    try:
        import eval.bleurt_eval_cli as bleurt_eval_cli
    except Exception as e:
        raise ImportError(f"BLEURT metric requested but bleurt_eval_cli not found: {e}")
    bleurt_path = bleurt_model_path if bleurt_model_path is not None else "BLEURT-20"
    bleurt_output = bleurt_eval_cli.func_call(bleurt_path, mt_flat, flat_ref)
    return _normalize_metric_output(bleurt_output, N, runs)

def run_oss_eval(
    df: pd.DataFrame,
    mt_flat: list[str],
    runs: int,
    oss_model,
    oss_model_path: Optional[str] = None,
) -> list[float]:
    """Run OSS evaluation on the concatenated dataset, single batched call.

    Args:
        df: concatenated DataFrame.
        mt_flat: flat predictions, length ``runs * len(df)``, run-major order.
        runs: number of inference runs.
        oss_model: loaded vLLM model for OSS.
        oss_model_path: path to the OSS model checkpoint.

    Returns:
        Flat OSS scores, length ``runs * len(df)``, run-major order.
    """
    N = len(df)
    if N == 0:
        raise ValueError("Input data is empty: df has 0 rows")

    ref_list = df["trg_text"].tolist()
    src_list = df["src_text"].tolist()
    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()

    # Handle comment column per-row
    if "comment" in df.columns:
        comment_list = df["comment"].tolist()
        ref_for_oss = [
            f"{ref}\n评估重点：\n{comment}" if pd.notna(comment) else ref
            for ref, comment in zip(ref_list, comment_list)
        ]
    else:
        ref_for_oss = ref_list

    flat_ref = ref_for_oss * runs
    flat_src = src_list * runs
    flat_src_langs = src_langs * runs
    flat_trg_langs = trg_langs * runs

    model_path = oss_model_path if oss_model_path is not None else "openai/gpt-oss-120b"
    oss_output = run_oss_SQM.func_call(
        src_list=flat_src,
        mt_list=mt_flat,
        src_langs=flat_src_langs,
        trg_langs=flat_trg_langs,
        ref_list=flat_ref,
        model=oss_model,
        model_path=model_path,
    )
    return _normalize_metric_output(oss_output, N, runs)


def _clear_mem():
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def main(
    data_id: tuple[str],
    model_path: str,
    model_name: str,
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    metrics: list[str] = ["bleurt", "oss"],
    prompt_type: str = "codeblock-think",
    runs: int = 1,
    save_results: bool = False,
    difficulty_filter: int = 0,
    data_dir: Optional[str] = None,
    **kwargs,
):
    """
    Run machine translation evaluation for a model on specified datasets.

    All datasets are concatenated and processed in single batched calls for both
    inference and evaluation, then results are split back per dataset for reporting.

    Args:
        data_id: One or more dataset identifiers from MT_TEST_DATA_META_INFO.
            Can be a single string (comma-separated), tuple, or iterable of dataset IDs.
        model_path: Path to the pretrained MT model weights.
        model_name: Name of the model for logging and output file naming.
        temperature: Sampling temperature for generation. Defaults to 0.4.
        top_p: Nucleus sampling probability threshold. Defaults to 0.7.
        max_new_tokens: Maximum number of tokens to generate per translation.
            Defaults to 4096.
        metrics: List of evaluation metrics to compute. Supported: 'bleurt', 'oss'.
            Defaults to ["bleurt", "oss"].
        prompt_type: Type of prompt template. Defaults to "codeblock-think".
        runs: Number of inference runs per sample. Defaults to 1.
        save_results: Whether to save per-item results to JSON. Defaults to False.
        difficulty_filter: Minimum difficulty score for notes to be used. Items with
            difficulty below this threshold will have notes set to None. Defaults to 0.
        **kwargs: Additional keyword arguments:
            - bleurt_model_path: Path to BLEURT model.
            - oss_model_path: Path to gpt-oss model.
            - mt_vllm_kwargs: vLLM kwargs for the MT model.
            - oss_vllm_kwargs: vLLM kwargs for the OSS model.
    """
    # Parse data_id input
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

    # Load and concatenate all datasets
    if data_dir:
        df_all, boundaries, dfs_per_id = load_datasets_from_dir(data_id_list, data_dir)
        lang_pairs = {did: "unknown" for did in data_id_list}
    else:
        df_all, boundaries, lang_pairs, dfs_per_id = _load_datasets(data_id_list)
    N = len(df_all)

    # Build notes list
    flat_notes_list, flat_use_notes_mask = build_notes_list(
        df_all, difficulty_filter, runs
    )
    notes_used = sum(flat_use_notes_mask[:N])
    print(f"Total items: {N}, notes used: {notes_used}/{N} (difficulty_filter={difficulty_filter})")

    # Stage 1: Inference — single batched call across all data_ids and runs
    mt_vllm_kwargs = kwargs.get("mt_vllm_kwargs", {})
    model, tokenizer = load_model_tokenizer(model_path, **mt_vllm_kwargs)

    mt_flat = run_inference(
        df_all,
        model,
        tokenizer,
        model_path,
        temperature,
        top_p,
        max_new_tokens,
        prompt_type=prompt_type,
        runs=runs,
        notes_list=flat_notes_list,
    )

    # Release MT model to free GPU memory
    try:
        del model
        del tokenizer
        _clear_mem()
    except Exception:
        pass

    # Extract model paths from kwargs
    bleurt_model_path = kwargs.get("bleurt_model_path")
    oss_model_path = kwargs.get("oss_model_path")

    # Stage 2: Evaluation — single batched call per metric
    datasets_metric_results: Dict[str, Dict[str, float]] = {did: {} for did in data_id_list}
    datasets_metric_none_counts: Dict[str, Dict[str, int]] = {did: {} for did in data_id_list}
    datasets_per_item_metric_avgs: Dict[str, Dict[str, list[Optional[float]]]] = {
        did: {} for did in data_id_list
    }
    datasets_valid_metrics: Dict[str, List[str]] = {did: [] for did in data_id_list}
    datasets_metrics_by_notes: Dict[str, Dict[str, dict]] = {did: {} for did in data_id_list}

    all_valid_metrics: List[str] = []
    seen_metrics = set()

    # OSS evaluation
    if "oss" in metrics:
        if oss_model_path is None:
            oss_model_path = "openai/gpt-oss-120b"
        oss_vllm_kwargs = kwargs.get("oss_vllm_kwargs", {})
        oss_model = run_oss_SQM.init_oss_model(oss_model_path, **oss_vllm_kwargs)

        oss_scores_flat = run_oss_eval(
            df_all,
            mt_flat,
            runs,
            oss_model,
            oss_model_path=oss_model_path,
        )
        oss_notes_split = split_metrics_by_notes(
            oss_scores_flat, flat_use_notes_mask, boundaries, N, runs
        )

        for did in data_id_list:
            datasets_metric_results[did]["oss"] = oss_notes_split[did]["all"]["avg"]
            datasets_metric_none_counts[did]["oss"] = oss_notes_split[did]["all"]["none_count"]
            datasets_per_item_metric_avgs[did]["oss"] = oss_notes_split[did]["all"]["per_item_avgs"]
            datasets_metrics_by_notes[did]["oss"] = oss_notes_split[did]
            datasets_valid_metrics[did].append("oss")

        if "oss" not in seen_metrics:
            seen_metrics.add("oss")
            all_valid_metrics.append("oss")

        try:
            del oss_model
            _clear_mem()
        except Exception:
            pass

    # BLEURT evaluation
    if "bleurt" in metrics:
        bleurt_scores_flat = run_bleurt_eval(
            df_all,
            mt_flat,
            runs,
            bleurt_model_path=bleurt_model_path,
        )
        bleurt_notes_split = split_metrics_by_notes(
            bleurt_scores_flat, flat_use_notes_mask, boundaries, N, runs
        )

        for did in data_id_list:
            datasets_metric_results[did]["bleurt"] = bleurt_notes_split[did]["all"]["avg"]
            datasets_metric_none_counts[did]["bleurt"] = bleurt_notes_split[did]["all"]["none_count"]
            datasets_per_item_metric_avgs[did]["bleurt"] = bleurt_notes_split[did]["all"]["per_item_avgs"]
            datasets_metrics_by_notes[did]["bleurt"] = bleurt_notes_split[did]
            datasets_valid_metrics[did].append("bleurt")

        if "bleurt" not in seen_metrics:
            seen_metrics.add("bleurt")
            all_valid_metrics.append("bleurt")

    # Print summary
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

    # Optionally save per-data_id results
    if save_results:
        for did, (start, end) in boundaries.items():
            n = end - start
            mt_nested = [
                [mt_flat[r * N + start + i] for i in range(n)]
                for r in range(runs)
            ]
            notes_per_item = flat_notes_list[start:end]
            mask_per_item = flat_use_notes_mask[start:end]
            save_results_to_json(
                df=dfs_per_id[did],
                mt_list_for_runs_nested=mt_nested,
                per_item_metric_avgs=datasets_per_item_metric_avgs[did],
                valid_metrics=datasets_valid_metrics[did],
                dataset_name=did,
                model_name=model_name,
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                runs=runs,
                prompt_type=prompt_type,
                difficulty_filter=difficulty_filter,
                notes_list_per_item=notes_per_item,
                use_notes_mask_per_item=mask_per_item,
            )

    wandb_config = {
        "dataset_names": data_id_list,
        "model_path": model_path,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "runs": runs,
        "metrics": all_valid_metrics,
        "lang_pairs": lang_pairs,
        "prompt_type": prompt_type,
        "difficulty_filter": difficulty_filter,
        "data_dir": data_dir,
    }

    log_results_to_wandb(
        datasets_metric_results=datasets_metric_results,
        config=wandb_config,
        datasets_metric_none_counts=datasets_metric_none_counts,
        datasets_metrics_by_notes=datasets_metrics_by_notes,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
