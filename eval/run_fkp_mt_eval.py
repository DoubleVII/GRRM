"""Evaluate one-pass local FKP models using the existing MT metrics."""

import json
from pathlib import Path
from typing import Optional

from eval.run_mt_ffgpe_eval import save_ffgpe_results_by_dataset
from eval.run_mt_eval import (
    _load_datasets, _release_vllm_model, _sanitize_filename_component,
    _split_scores_by_data_id, log_results_to_wandb, run_bleurt_eval,
    run_oss_eval, run_oss_SQM,
)
from inference.run_fkp_mt import func_call
from inference.run_mt import load_model_tokenizer
from utils.helpers import load_datasets_from_dir


def main(
    data_id,
    model_path: str,
    model_name: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 8192,
    metrics: list[str] = ("bleurt", "oss"),
    runs: int = 1,
    save_results: bool = False,
    data_dir: Optional[str] = None,
    retry: int = 3,
    relaxed_parsing: bool = False,
    **kwargs,
):
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if retry < 0:
        raise ValueError("retry must be non-negative")
    if isinstance(metrics, str):
        metrics = [value.strip() for value in metrics.split(",") if value.strip()]
    if any(metric not in {"bleurt", "oss"} for metric in metrics):
        raise ValueError("Supported metrics are bleurt and oss")
    data_ids = tuple(x.strip() for x in data_id.split(",") if x.strip()) if isinstance(data_id, str) else tuple(data_id)
    if not data_ids:
        raise ValueError("data_id must contain at least one dataset")
    if data_dir:
        df, boundaries, per_id = load_datasets_from_dir(data_ids, data_dir)
    else:
        df, boundaries, _, per_id = _load_datasets(data_ids)
    n_items = len(df)
    if not n_items:
        raise ValueError("Evaluation requires at least one sample")
    model, tokenizer = load_model_tokenizer(model_path, **kwargs.get("mt_vllm_kwargs", {}))
    try:
        output = func_call(
            model_path, df.src_text.tolist() * runs, df.src_lang.tolist() * runs,
            df.trg_lang.tolist() * runs, model=model, tokenizer=tokenizer,
            temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, retry=retry,
            relaxed_parsing=relaxed_parsing,
        )
    finally:
        _release_vllm_model(model)
    predictions = output["responses"]
    if len(predictions) != n_items * runs:
        raise ValueError(f"Expected {n_items * runs} predictions, got {len(predictions)}")
    # Drop Python references too before loading the OSS evaluator on these GPUs.
    del model, tokenizer
    metric_results = {did: {} for did in data_ids}
    metric_none = {did: {} for did in data_ids}
    per_item = {did: {} for did in data_ids}
    valid_metrics = {did: [] for did in data_ids}
    for metric in metrics:
        if metric == "oss":
            oss_path = kwargs.get("oss_model_path", "openai/gpt-oss-120b")
            oss_model = run_oss_SQM.init_oss_model(oss_path, **kwargs.get("oss_vllm_kwargs", {}))
            try:
                scores = run_oss_eval(df, predictions, runs, oss_model, oss_path)
            finally:
                _release_vllm_model(oss_model)
            del oss_model
        else:
            scores = run_bleurt_eval(df, predictions, runs, kwargs.get("bleurt_model_path"))
        split = _split_scores_by_data_id(scores, boundaries, n_items, runs)
        for did in data_ids:
            metric_results[did][metric] = split[did]["avg"]
            metric_none[did][metric] = split[did]["none_count"]
            per_item[did][metric] = split[did]["per_item_avgs"]
            valid_metrics[did].append(metric)
    counts = [count for count, valid in zip(output["key_point_counts"], output["parser_valid"]) if valid]
    summary = {
        "method": "fkp_mt", "protocol": "simple",
        "key_point_count_mean": sum(counts) / len(counts) if counts else None,
        "key_point_count_distribution": {str(count): counts.count(count) for count in sorted(set(counts))},
        "items_without_key_points": counts.count(0),
        "parser_failures": sum(not valid for valid in output["parser_valid"]),
        "generation_failures": predictions.count("Translation Failed."),
    }
    print(json.dumps({"summary": summary, "metrics": metric_results, "metric_none": metric_none}, indent=2))
    settings = {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens,
                "retry": retry, "relaxed_parsing": relaxed_parsing}
    if save_results:
        extra_items = [{key: [output[field][r * n_items + i] for r in range(runs)]
                        for key, field in (("key_points", "key_points"), ("key_point_count", "key_point_counts"),
                                           ("raw_output", "raw_outputs"), ("parsed_output", "parsed_outputs"),
                                           ("parser_valid", "parser_valid"))}
                       for i in range(n_items)]
        save_ffgpe_results_by_dataset(
            df=df, boundaries=boundaries, per_id=per_id, data_ids=data_ids,
            predictions=predictions, runs=runs, metric_results=metric_results,
            per_item_metrics=per_item, valid_metrics=valid_metrics, model_name=model_name,
            model_path=model_path, prompt_type="fkp_mt", extra_items=extra_items,
            metadata={"method": "fkp_mt", "protocol": "simple", "summary": summary, "settings": settings},
        )
        items = []
        for i in range(n_items):
            did = next(did for did, (start, end) in boundaries.items() if start <= i < end)
            local_index = i - boundaries[did][0]
            items.append({"data_id": did, "index": local_index, "source": df.iloc[i].src_text,
                          "reference": df.iloc[i].trg_text,
                          "predictions": [predictions[r * n_items + i] for r in range(runs)],
                          "metrics_avg": {metric: per_item[did][metric][local_index] for metric in metrics},
                          **extra_items[i]})
        payload = {"method": "fkp_mt", "protocol": "simple", "model_name": model_name,
                   "model_path": model_path, "data_name": list(data_ids), "runs": runs,
                   "settings": settings, "metrics": metric_results, "summary": summary, "items": items}
        Path(f"{_sanitize_filename_component(model_name)}__fkp_mt.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_results_to_wandb(metric_results, {"model_name": model_name, "model_path": model_path,
                         "metrics": list(metrics), "method": "fkp_mt", "protocol": "simple", **settings}, metric_none)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
