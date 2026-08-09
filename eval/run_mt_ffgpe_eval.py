"""Evaluation entry point for the standalone one-pass FFGPE method."""

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from utils.config import MT_TEST_DATA_META_INFO
from utils.helpers import load_datasets_from_dir
from inference.run_mt import load_model_tokenizer
from inference.run_mt_ffgpe import func_call
from eval.run_mt_eval import (
    _load_datasets,
    _release_vllm_model,
    log_results_to_wandb,
    run_bleurt_eval,
    run_oss_eval,
    run_oss_SQM,
)


def main(
    data_id,
    model_path: str,
    model_name: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 8192,
    metrics: list[str] = ["bleurt", "oss"],
    prompt_type: str = "fixed_4",
    max_candidates: int = 4,
    runs: int = 1,
    save_results: bool = False,
    data_dir: Optional[str] = None,
    retry: int = 3,
    **kwargs,
):
    data_ids = (data_id,) if isinstance(data_id, str) else tuple(data_id)
    if len(data_ids) == 1 and isinstance(data_ids[0], str) and "," in data_ids[0]:
        data_ids = tuple(x.strip() for x in data_ids[0].split(",") if x.strip())
    if not data_ids:
        raise ValueError("data_id must contain at least one dataset")
    if data_dir:
        df, boundaries, per_id = load_datasets_from_dir(data_ids, data_dir)
        lang_pairs = {x: "unknown" for x in data_ids}
    else:
        df, boundaries, lang_pairs, per_id = _load_datasets(data_ids)
    n_items = len(df)
    model, tokenizer = load_model_tokenizer(model_path, **kwargs.get("mt_vllm_kwargs", {}))
    flat_src = df.src_text.tolist() * runs
    flat_src_lang = df.src_lang.tolist() * runs
    flat_trg_lang = df.trg_lang.tolist() * runs
    output = func_call(
        model_path=model_path,
        src_list=flat_src,
        src_langs=flat_src_lang,
        trg_langs=flat_trg_lang,
        max_candidates=max_candidates,
        prompt_type=prompt_type,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        retry=retry,
        model=model,
        tokenizer=tokenizer,
    )
    predictions = output["responses"]
    expected = n_items * runs
    if len(predictions) != expected:
        raise ValueError(f"Expected {expected} predictions, got {len(predictions)}")
    try:
        _release_vllm_model(model)
    except Exception:
        pass

    metric_results = {did: {} for did in data_ids}
    metric_none = {did: {} for did in data_ids}
    per_item = {did: {} for did in data_ids}
    valid_metrics = {did: [] for did in data_ids}
    all_metrics = []
    for metric in metrics:
        if metric == "oss":
            oss_path = kwargs.get("oss_model_path", "openai/gpt-oss-120b")
            oss_model = run_oss_SQM.init_oss_model(oss_path, **kwargs.get("oss_vllm_kwargs", {}))
            scores = run_oss_eval(df, predictions, runs, oss_model, oss_path)
            _release_vllm_model(oss_model)
        elif metric == "bleurt":
            scores = run_bleurt_eval(df, predictions, runs, kwargs.get("bleurt_model_path"))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        from eval.run_mt_eval import _split_scores_by_data_id
        split = _split_scores_by_data_id(scores, boundaries, n_items, runs)
        for did in data_ids:
            metric_results[did][metric] = split[did]["avg"]
            metric_none[did][metric] = split[did]["none_count"]
            per_item[did][metric] = split[did]["per_item_avgs"]
            valid_metrics[did].append(metric)
        all_metrics.append(metric)

    counts = output.get("candidate_counts", [])
    summary = {
        "method": "ffgpe",
        "candidate_count_mean": (sum(counts) / len(counts)) if counts else 0.0,
        "candidate_count_min": min(counts) if counts else 0,
        "candidate_count_max": max(counts) if counts else 0,
        "candidate_count_distribution": {str(x): counts.count(x) for x in sorted(set(counts))},
        "parser_failures": sum(not x for x in output.get("parser_valid", [])),
        "generation_failures": sum(x == "Translation Failed." for x in predictions),
    }
    print("\n=== FFGPE evaluation summary ===")
    print(f"method: ffgpe | model: {model_name} | prompt_type: {prompt_type} | max_candidates: {max_candidates} | runs: {runs}")
    print(
        "candidate_count: "
        f"mean={summary['candidate_count_mean']:.2f}, "
        f"range=[{summary['candidate_count_min']}, {summary['candidate_count_max']}], "
        f"distribution={summary['candidate_count_distribution']}"
    )
    print(
        f"failures: parser={summary['parser_failures']}, "
        f"generation={summary['generation_failures']}"
    )
    for did in data_ids:
        print(f"\n--- {did} ---")
        for metric, value in metric_results[did].items():
            none_count = metric_none[did].get(metric, 0)
            print(f"{metric}: {value:.6f} (none={none_count})")
    if save_results:
        payload = {
            "method": "ffgpe", "data_name": list(data_ids), "model_name": model_name,
            "model_path": model_path, "prompt_type": prompt_type,
            "max_candidates": max_candidates, "metrics": metric_results,
            "summary": summary, "items": [],
        }
        for i in range(n_items):
            payload["items"].append({
                "source": df.iloc[i].src_text, "reference": df.iloc[i].trg_text,
                "predictions": [predictions[r * n_items + i] for r in range(runs)],
                "candidates": [output.get("candidates", [])[r * n_items + i] for r in range(runs)],
                "candidate_count": [counts[r * n_items + i] for r in range(runs)],
                "raw_output": [output.get("raw_outputs", [])[r * n_items + i] for r in range(runs)],
                "parser_valid": [output.get("parser_valid", [])[r * n_items + i] for r in range(runs)],
            })
        Path(f"{model_name}__ffgpe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_results_to_wandb(metric_results, {"model_name": model_name, "model_path": model_path, "metrics": all_metrics, "method": "ffgpe", "prompt_type": prompt_type, "max_candidates": max_candidates}, metric_none)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
