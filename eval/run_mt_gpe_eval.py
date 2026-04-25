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
from inference.run_mt import load_model_tokenizer
from inference.run_gpe import func_call as gpe_func_call
import inference.run_mt as run_mt
from utils.config import MT_TEST_DATA_META_INFO


def _load_datasets_from_dir(
    data_id_list: tuple[str, ...],
    data_dir: str,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]], dict[str, pd.DataFrame]]:
    """Load and concatenate parquets from ``{data_dir}/{data_id}.parquet``.

    Returns:
        df_all: concatenated DataFrame with a ``_data_id`` column.
        boundaries: ``{data_id: (start_idx, end_idx)}`` into df_all rows.
        dfs_per_id: ``{data_id: original DataFrame}``.
    """
    base = Path(data_dir)
    frames: list[pd.DataFrame] = []
    boundaries: dict[str, tuple[int, int]] = {}
    dfs_per_id: dict[str, pd.DataFrame] = {}

    offset = 0
    for did in data_id_list:
        p = base / f"{did}.parquet"
        if not p.exists():
            raise ValueError(f"Data file not found: {p}")

        df = pd.read_parquet(p)
        df["_data_id"] = did
        n = len(df)
        boundaries[did] = (offset, offset + n)
        dfs_per_id[did] = df

        frames.append(df)
        offset += n

    df_all = pd.concat(frames, ignore_index=True)
    return df_all, boundaries, dfs_per_id


def _build_notes_list(
    df: pd.DataFrame,
    difficulty_filter: int = 0,
    runs: int = 1,
) -> tuple[list[Optional[str]], list[bool]]:
    """Build notes list with difficulty filtering.

    Returns:
        flat_notes_list: length ``runs * len(df)``, run-major order.
            Items below difficulty_filter have notes set to None.
        flat_use_notes_mask: length ``runs * len(df)``. True where notes
            were actually provided to GPE.
    """
    has_notes = "notes" in df.columns
    has_difficulty = "difficulty" in df.columns

    per_item_notes: list[Optional[str]] = []
    per_item_use_notes: list[bool] = []

    for _, row in df.iterrows():
        notes: Optional[str] = None
        difficulty = 0

        if has_notes:
            raw = row["notes"]
            if isinstance(raw, str) and raw.strip():
                notes = raw

        if has_difficulty:
            d = row.get("difficulty", 0)
            difficulty = d if pd.notna(d) else 0

        use_notes = notes is not None
        if difficulty_filter > 0 and difficulty < difficulty_filter:
            use_notes = False

        per_item_notes.append(notes if use_notes else None)
        per_item_use_notes.append(use_notes)

    flat_notes_list = per_item_notes * runs
    flat_use_notes_mask = per_item_use_notes * runs
    return flat_notes_list, flat_use_notes_mask


def _split_metrics_by_notes(
    scores_flat: list[float],
    use_notes_mask_flat: list[bool],
    boundaries: dict[str, tuple[int, int]],
    total_n: int,
    runs: int,
) -> dict[str, dict[str, dict]]:
    """Split flat scores by data_id and by notes/no-notes usage.

    Returns:
        ``{data_id: {
            "all":      {"avg", "none_count", "per_item_avgs", "count"},
            "notes":    {"avg", "none_count", "count"},
            "no_notes": {"avg", "none_count", "count"},
        }}``
    """
    results: dict[str, dict[str, dict]] = {}

    for did, (start, end) in boundaries.items():
        n = end - start

        all_scores: list[float] = []
        notes_scores: list[float] = []
        no_notes_scores: list[float] = []
        notes_item_count = 0

        for r in range(runs):
            base = r * total_n
            for i in range(start, end):
                idx = base + i
                score = scores_flat[idx]
                all_scores.append(score)
                if use_notes_mask_flat[idx]:
                    notes_scores.append(score)
                else:
                    no_notes_scores.append(score)

        # Count unique items with notes (from first run)
        for i in range(start, end):
            if use_notes_mask_flat[i]:
                notes_item_count += 1

        avg_all, nc_all = _average_overall(all_scores)
        per_item_avgs = _average_per_item(all_scores, n, runs)
        avg_notes, nc_notes = _average_overall(notes_scores)
        avg_no, nc_no = _average_overall(no_notes_scores)

        results[did] = {
            "all": {
                "avg": avg_all,
                "none_count": nc_all,
                "per_item_avgs": per_item_avgs,
                "count": n,
            },
            "notes": {
                "avg": avg_notes,
                "none_count": nc_notes,
                "count": notes_item_count,
            },
            "no_notes": {
                "avg": avg_no,
                "none_count": nc_no,
                "count": n - notes_item_count,
            },
        }
    return results


def log_gpe_results_to_wandb(
    datasets_metrics_by_notes: Dict[str, Dict[str, dict]],
    valid_metrics: List[str],
    config: Dict[str, Any],
    datasets_metric_none_counts: Optional[Dict[str, Dict[str, int]]] = None,
):
    """Log GPE results to wandb with overall + notes/no-notes breakdown.

    Args:
        datasets_metrics_by_notes: ``{data_id: {metric: {"all", "notes", "no_notes"}}}``
        valid_metrics: ordered list of metric names.
        config: wandb config dict.
        datasets_metric_none_counts: ``{data_id: {metric: none_count}}`` (for "all" group).
    """
    project_name = "mt-gpe-eval"
    wandb.init(project=project_name, name=config["model_name"], config=config)

    # Build table: data_id, {metric}/all, {metric}/notes, {metric}/no_notes, notes_count, no_notes_count
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
        # counts from first metric (same across metrics)
        first_split = next(iter(metric_dict.values()), {})
        notes_count = first_split.get("notes", {}).get("count", 0)
        no_notes_count = first_split.get("no_notes", {}).get("count", 0)
        row.append(notes_count)
        row.append(no_notes_count)
        rows.append(row)

    table = wandb.Table(columns=columns, data=rows)
    wandb.log({"metrics_by_dataset": table})

    # Sync to summary
    for data_id, metric_dict in datasets_metrics_by_notes.items():
        for m, split in metric_dict.items():
            for group in ("all", "notes", "no_notes"):
                key = f"{data_id}/{m}/{group}"
                val = split.get(group, {}).get("avg", np.nan)
                wandb.run.summary[key] = val
        # counts
        first_split = next(iter(metric_dict.values()), {})
        wandb.run.summary[f"{data_id}/notes_count"] = first_split.get("notes", {}).get("count", 0)
        wandb.run.summary[f"{data_id}/no_notes_count"] = first_split.get("no_notes", {}).get("count", 0)

    # Log none counts
    if datasets_metric_none_counts:
        for data_id, metric_none_counts in datasets_metric_none_counts.items():
            for m, cnt in metric_none_counts.items():
                key = f"none_count/{data_id}/{m}"
                try:
                    wandb.run.summary[key] = int(cnt)
                except Exception:
                    wandb.run.summary[key] = cnt


def save_gpe_results_to_json(
    df: pd.DataFrame,
    pe_nested: list[list[str]],
    mt_candidates_first_run: list[list[str]],
    per_item_metric_avgs: Dict[str, list[Optional[float]]],
    valid_metrics: list[str],
    notes_list_per_item: list[Optional[str]],
    use_notes_mask_per_item: list[bool],
    difficulty_list_per_item: list,
    dataset_name: str,
    model_name: str,
    model_path: str,
    gpe_model_path: str,
    sampling_n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    runs: int,
    prompt_type: str,
    difficulty_filter: int,
) -> Path:
    safe_model_name = _sanitize_filename_component(model_name)
    safe_dataset_name = _sanitize_filename_component(dataset_name)
    out_file = Path.cwd() / f"{safe_model_name}__gpe__{safe_dataset_name}.json"

    n = len(df)
    items = []
    for i in range(n):
        preds = [pe_nested[r][i] for r in range(runs)]
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
            "predictions": preds,
            "notes": notes_list_per_item[i],
            "use_notes": use_notes_mask_per_item[i],
            "difficulty": difficulty_list_per_item[i],
            "metrics_avg": metrics_avg_item,
        })

    json_payload = {
        "data_name": dataset_name,
        "model_name": model_name,
        "model_path": model_path,
        "gpe_model_path": gpe_model_path,
        "sampling_n": sampling_n,
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


def main(
    data_id: tuple[str],
    model_path: str,
    gpe_model_path: str,
    model_name: str,
    data_dir: Optional[str] = None,
    sampling_n: int = 4,
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    metrics: list[str] = ["bleurt", "oss"],
    prompt_type: str = "codeblock-think",
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
    notes_used = sum(flat_use_notes_mask[:N])  # count from first run
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
    # When sampling_n == 1, run_mt returns a flat list[str]; wrap for GPE
    if sampling_n == 1:
        mt_responses = [[r] for r in mt_responses]

    # --- Conditionally release MT model ---
    if gpe_model_path == model_path:
        gpe_model, gpe_tokenizer = model, tokenizer
    else:
        del model, tokenizer
        _clear_mem()
        gpe_model, gpe_tokenizer = None, None

    # --- Stage 2: GPE inference ---
    print(f"Running GPE inference: {runs * N} items ...")
    if gpe_model is None:
        gpe_vllm_kwargs = kwargs.get("gpe_vllm_kwargs", {})
        gpe_model, gpe_tokenizer = load_model_tokenizer(gpe_model_path, **gpe_vllm_kwargs)

    gpe_output = gpe_func_call(
        model_path=gpe_model_path,
        src_list=flat_src,
        mt_list=mt_responses,
        src_langs=flat_src_langs,
        trg_langs=flat_trg_langs,
        notes_list=flat_notes_list,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        model=gpe_model,
        tokenizer=gpe_tokenizer,
    )
    pe_flat = gpe_output["post_edit_mt"]

    # Release GPE model
    try:
        del gpe_model, gpe_tokenizer
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

    # OSS evaluation
    if "oss" in metrics:
        if oss_model_path is None:
            oss_model_path = "openai/gpt-oss-120b"
        import inference.run_oss_SQM as run_oss_SQM

        oss_vllm_kwargs = kwargs.get("oss_vllm_kwargs", {})
        oss_model = run_oss_SQM.init_oss_model(oss_model_path, **oss_vllm_kwargs)

        oss_scores_flat = run_oss_eval(
            df_all, pe_flat, runs, oss_model, oss_model_path=oss_model_path
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

    # BLEURT evaluation
    if "bleurt" in metrics:
        bleurt_scores_flat = run_bleurt_eval(
            df_all, pe_flat, runs, bleurt_model_path=bleurt_model_path
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
            pe_nested = [
                [pe_flat[r * N + start + i] for i in range(n)]
                for r in range(runs)
            ]
            # MT candidates from first run
            mt_candidates_first_run = [mt_responses[start + i] for i in range(n)]
            # Notes metadata per item (from the original df, not tiled)
            notes_per_item = flat_notes_list[start:end]
            mask_per_item = flat_use_notes_mask[start:end]
            difficulty_per_item = df_all.iloc[start:end].get("difficulty", [0] * n).tolist()

            save_gpe_results_to_json(
                df=dfs_per_id[did],
                pe_nested=pe_nested,
                mt_candidates_first_run=mt_candidates_first_run,
                per_item_metric_avgs=datasets_per_item_metric_avgs[did],
                valid_metrics=datasets_valid_metrics[did],
                notes_list_per_item=notes_per_item,
                use_notes_mask_per_item=mask_per_item,
                difficulty_list_per_item=difficulty_per_item,
                dataset_name=did,
                model_name=model_name,
                model_path=model_path,
                gpe_model_path=gpe_model_path,
                sampling_n=sampling_n,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                runs=runs,
                prompt_type=prompt_type,
                difficulty_filter=difficulty_filter,
            )

    # --- Wandb logging ---
    wandb_config = {
        "dataset_names": data_id_list,
        "model_path": model_path,
        "gpe_model_path": gpe_model_path,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "runs": runs,
        "sampling_n": sampling_n,
        "metrics": all_valid_metrics,
        "lang_pairs": lang_pairs,
        "prompt_type": prompt_type,
        "difficulty_filter": difficulty_filter,
        "data_dir": data_dir,
    }

    log_gpe_results_to_wandb(
        datasets_metrics_by_notes=datasets_metrics_by_notes,
        valid_metrics=all_valid_metrics,
        config=wandb_config,
        datasets_metric_none_counts=datasets_metric_none_counts,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
