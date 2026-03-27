import pandas as pd
from pathlib import Path
import json

from typing import Optional, Dict, Any, List, Iterable
import wandb

from utils.config import MT_TEST_DATA_META_INFO
import numpy as np
from inference.run_mt import load_model_tokenizer
import inference.run_oss_SQM as run_oss_SQM

def log_results_to_wandb(
    datasets_metric_results: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    datasets_metric_none_counts: Optional[Dict[str, Dict[str, int]]] = None,
):
    """Log results from multiple datasets to wandb as a table.

    Args:
        datasets_metric_results: {dataset_name: {metric: value}}
        config: wandb config
        datasets_metric_none_counts: {dataset_name: {metric: none_count}}
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
        "metrics": valid_metrics,
        "items": items,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    return out_file



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
):
    import inference.run_mt as run_mt

    src_list = df["src_text"].tolist()
    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()

    mt_list_for_runs = []
    for _ in range(runs):        
        func_call_kwargs = {
            "model_path": model_path,
            "src_list": src_list,
            "src_langs": src_langs,
            "trg_langs": trg_langs,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "prompt_type": prompt_type,
            "model": model,
            "tokenizer": tokenizer,
        }
        if "Seed-X" in model_path:
            func_call_kwargs["use_chat_template"] = False
        output_dict = run_mt.func_call(**func_call_kwargs)
        mt_list = output_dict["responses"]
        if len(mt_list) != len(df):
            raise ValueError(
                f"mt_list must have the same length as src_list, but got {len(mt_list)} and {len(df)}"
            )

        mt_list_for_runs.append(mt_list)

    return mt_list_for_runs


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

def _average_overall(scores: list[float]) -> tuple[float, int]:
    vals: list[float] = []
    none_count: int = 0
    for s in scores:
        if s is None:
            none_count += 1
            continue
        try:
            vals.append(float(s))
        except Exception:
            none_count += 1
            continue
    if not vals:
        return float("nan"), none_count
    return sum(vals) / len(vals), none_count

def _average_per_item(scores: list[float], n_items: int, n_runs: int) -> list[Optional[float]]:
    avgs: list[Optional[float]] = []
    for i in range(n_items):
        vals: list[float] = []
        for r in range(n_runs):
            idx = r * n_items + i
            s = scores[idx]
            if s is None:
                continue
            try:
                vals.append(float(s))
            except Exception:
                continue
        if vals:
            avgs.append(sum(vals) / len(vals))
        else:
            avgs.append(None)
    return avgs

def run_bleurt_eval(
    df: pd.DataFrame,
    mt_list_for_runs: list[list[str]],
    runs: int,
    bleurt_model_path: Optional[str] = None,
) -> tuple[float, int, list[Optional[float]]]:
    mt_list_for_runs = [item for sublist in mt_list_for_runs for item in sublist]

    n = len(df)
    if n == 0:
        raise ValueError("Input data is empty: df has 0 rows")

    ref_mt_list = df["trg_text"].tolist()
    refs_flat = ref_mt_list * runs

    try:
        import eval.bleurt_eval_cli as bleurt_eval_cli
    except Exception as e:
        raise ImportError(f"BLEURT metric requested but bleurt_eval_cli not found: {e}")
    bleurt_path = bleurt_model_path if bleurt_model_path is not None else "BLEURT-20"
    bleurt_output = bleurt_eval_cli.func_call(bleurt_path, mt_list_for_runs, refs_flat)
    bleurt_scores_flat = _normalize_metric_output(bleurt_output, n, runs)
    avg, none_count = _average_overall(bleurt_scores_flat)
    per_item_metric_avgs = _average_per_item(bleurt_scores_flat, n, runs)

    return avg, none_count, per_item_metric_avgs

def run_oss_eval(
    df: pd.DataFrame,
    mt_list_for_runs: list[list[str]],
    runs: int,
    oss_model,
    oss_model_path: Optional[str] = None,
) -> tuple[float, int, list[Optional[float]]]:
    mt_list_for_runs = [item for sublist in mt_list_for_runs for item in sublist]

    n = len(df)
    if n == 0:
        raise ValueError("Input data is empty: df has 0 rows")

    ref_mt_list = df["trg_text"].tolist()
    refs_flat = ref_mt_list * runs

    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()
    src_texts = df["src_text"].tolist()

    src_flat = src_texts * runs
    src_langs_flat = src_langs * runs
    trg_langs_flat = trg_langs * runs

    if "comment" in df.columns:
        comment_list = df["comment"].tolist()
        ref_hint_list = [f"{ref}\n评估重点：\n{comment}" for ref, comment in zip(refs_flat, comment_list)]
        ref_hint_flat = ref_hint_list * runs
    else:
        ref_hint_flat = None

    model_path = oss_model_path if oss_model_path is not None else "openai/gpt-oss-120b"
    oss_output = run_oss_SQM.func_call(
        src_list=src_flat,
        mt_list=mt_list_for_runs,
        src_langs=src_langs_flat,
        trg_langs=trg_langs_flat,
        ref_list=ref_hint_flat if ref_hint_flat is not None else refs_flat,
        model=oss_model,
        model_path=model_path,
    )
    oss_scores_flat = _normalize_metric_output(oss_output, n, runs)
    avg, none_count = _average_overall(oss_scores_flat)
    per_item_metric_avgs = _average_per_item(oss_scores_flat, n, runs)

    return avg, none_count, per_item_metric_avgs


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
    **kwargs,
):
    """
    Run machine translation evaluation for a model on specified datasets.

    This function performs a two-stage evaluation pipeline:
    1. Inference stage: Load the MT model and generate translations for all datasets
    2. Evaluation stage: Release the MT model, load evaluation models (e.g., OSS),
       and compute metrics on the generated translations.

    Results are logged to Weights & Biases and optionally saved to JSON files.

    Args:
        data_id: One or more dataset identifiers from MT_TEST_DATA_META_INFO.
            Can be a single string (comma-separated), tuple, or iterable of dataset IDs.
        model_path: Path to the pretrained MT model weights.
        model_name: Name of the model for logging and output file naming.
        temperature: Sampling temperature for generation. Higher values produce more
            random outputs. Defaults to 0.4.
        top_p: Nucleus sampling probability threshold. Defaults to 0.7.
        max_new_tokens: Maximum number of tokens to generate per translation.
            Defaults to 4096.
        metrics: List of evaluation metrics to compute. Supported values are
            'bleurt' and 'oss'. The OSS model version can be specified via
            oss_model_path in kwargs. Defaults to ["bleurt", "oss"].
        prompt_type: Type of prompt template (and model output parser) to use for translation.
            Defaults to "codeblock-think".
        runs: Number of inference runs to perform for each sample. Multiple runs
            can be used to assess model consistency. Defaults to 1.
        save_results: Whether to save detailed results (predictions, per-item metrics)
            to JSON files in the current directory. Defaults to False.
        **kwargs: Additional keyword arguments. Supported keys:
            - bleurt_model_path: Path to the BLEURT model. If not provided, defaults to "BLEURT-20".
            - oss_model_path: Path to the gpt-oss model. If not provided, defaults to "openai/gpt-oss-120b".
            - mt_vllm_kwargs: Dictionary of parameters to pass to vLLM initialization for the MT model,
              such as gpu_memory_utilization, quantization, etc.
            - oss_vllm_kwargs: Dictionary of parameters to pass to vLLM initialization for the OSS model,
              such as gpu_memory_utilization, quantization, etc.

    Raises:
        ValueError: If data_id is empty or contains invalid dataset identifiers.
        ValueError: If the specified data path does not exist.
        ImportError: If BLEURT metric is requested but bleurt_eval_cli is not found.
    """
    if isinstance(data_id, Iterable):
        data_id_list = tuple(data_id)
    elif isinstance(data_id, str):
        data_id_list = tuple(data_id.strip().split(","))
    
    if isinstance(data_id, str):
        data_id_list = tuple(data_id.strip().split(","))
    elif isinstance(data_id, Iterable):
        data_id_list = tuple(data_id)
    else:
        data_id_list = tuple(data_id)
    
    if not data_id_list:
        raise ValueError(f"Invalid data_id. Please provide at least one valid data_id from {MT_TEST_DATA_META_INFO.keys()}")


    # Load MT model once and reuse, avoiding repeated loading in run_inference
    mt_vllm_kwargs = kwargs.get("mt_vllm_kwargs", {})
    model, tokenizer = load_model_tokenizer(model_path, **mt_vllm_kwargs)

    # Stage 1: Run translation inference on all datasets first, avoiding interleaving with evaluation (especially OSS)
    dfs: Dict[str, pd.DataFrame] = {}
    mt_lists_for_runs: Dict[str, list[list[str]]] = {}
    lang_pairs: Dict[str, str] = {}

    for did in data_id_list:
        if did not in MT_TEST_DATA_META_INFO:
            raise ValueError(
                f"data_id {did} not in MT_TEST_DATA_META_INFO: {MT_TEST_DATA_META_INFO.keys()}"
            )
        data_meta_info = MT_TEST_DATA_META_INFO[did]
        data_path = Path(data_meta_info["path"])
        lang_pair = f"{data_meta_info['src_lang']}2{data_meta_info['trg_lang']}"

        if not data_path.exists():
            raise ValueError(f"data_path {data_path} does not exist")

        df = pd.read_parquet(data_path)
        dfs[did] = df
        lang_pairs[did] = lang_pair

        mt_list_for_runs = run_inference(
            df,
            model,
            tokenizer,
            model_path,
            temperature,
            top_p,
            max_new_tokens,
            prompt_type=prompt_type,
            runs=runs,
        )
        mt_lists_for_runs[did] = mt_list_for_runs

    # MT inference stage complete, release MT model to free GPU memory for OSS evaluation
    try:
        del model
        del tokenizer
        _clear_mem()
    except Exception:
        pass

    # Extract model paths from kwargs
    bleurt_model_path = kwargs.get("bleurt_model_path")
    oss_model_path = kwargs.get("oss_model_path")

    # Stage 2: Evaluate by metric groups, first all bleurt, then all oss
    datasets_metric_results: Dict[str, Dict[str, float]] = {did: {} for did in data_id_list}
    datasets_metric_none_counts: Dict[str, Dict[str, int]] = {did: {} for did in data_id_list}
    datasets_per_item_metric_avgs: Dict[str, Dict[str, list[Optional[float]]]] = {did: {} for did in data_id_list}
    datasets_valid_metrics: Dict[str, List[str]] = {did: [] for did in data_id_list}

    all_valid_metrics: List[str] = []
    seen_metrics = set()

    # First run OSS evaluation for all datasets
    if "oss" in metrics:
        if oss_model_path is None:
            oss_model_path = "openai/gpt-oss-120b"
        oss_vllm_kwargs = kwargs.get("oss_vllm_kwargs", {})
        oss_model = run_oss_SQM.init_oss_model(oss_model_path, **oss_vllm_kwargs)

        for did in data_id_list:
            df = dfs[did]
            mt_list_for_runs = mt_lists_for_runs[did]

            avg, none_count, per_item_avgs = run_oss_eval(
                df,
                mt_list_for_runs,
                runs,
                oss_model,
                oss_model_path=oss_model_path,
            )

            datasets_metric_results[did]["oss"] = avg
            datasets_metric_none_counts[did]["oss"] = none_count
            datasets_per_item_metric_avgs[did]["oss"] = per_item_avgs
            datasets_valid_metrics[did].append("oss")

            if "oss" not in seen_metrics:
                seen_metrics.add("oss")
                all_valid_metrics.append("oss")

        # Release OSS model
        try:
            del oss_model
            _clear_mem()
        except Exception:
            pass

    # Then run BLEURT evaluation for all datasets
    if "bleurt" in metrics:
        for did in data_id_list:
            df = dfs[did]
            mt_list_for_runs = mt_lists_for_runs[did]

            avg, none_count, per_item_avgs = run_bleurt_eval(
                df,
                mt_list_for_runs,
                runs,
                bleurt_model_path=bleurt_model_path,
            )

            datasets_metric_results[did]["bleurt"] = avg
            datasets_metric_none_counts[did]["bleurt"] = none_count
            datasets_per_item_metric_avgs[did]["bleurt"] = per_item_avgs
            datasets_valid_metrics[did].append("bleurt")

            if "bleurt" not in seen_metrics:
                seen_metrics.add("bleurt")
                all_valid_metrics.append("bleurt")

    # Optionally save all inference results and per-item metric averages to current directory
    if save_results:
        for did in data_id_list:
            df = dfs[did]
            mt_list_for_runs = mt_lists_for_runs[did]
            valid_metrics = datasets_valid_metrics[did]
            per_item_metric_avgs = datasets_per_item_metric_avgs[did]

            save_results_to_json(
                df=df,
                mt_list_for_runs_nested=mt_list_for_runs,
                per_item_metric_avgs=per_item_metric_avgs,
                valid_metrics=valid_metrics,
                dataset_name=did,
                model_name=model_name,
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                runs=runs,
                prompt_type=prompt_type,
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
    }

    log_results_to_wandb(
        datasets_metric_results=datasets_metric_results,
        config=wandb_config,
        datasets_metric_none_counts=datasets_metric_none_counts,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
