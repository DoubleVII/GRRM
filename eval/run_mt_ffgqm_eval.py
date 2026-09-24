"""Evaluation entry point for the simple-protocol fused FlashGQM method."""

import json
from pathlib import Path
from typing import Optional

from utils.helpers import load_datasets_from_dir
from inference.run_mt import load_model_tokenizer
from inference.run_mt_ffgqm import func_call
from eval.run_mt_ffgpe_eval import save_ffgpe_results_by_dataset
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
    max_candidates: int = 4,
    runs: int = 1,
    save_results: bool = False,
    data_dir: Optional[str] = None,
    retry: int = 3,
    **kwargs,
):
    if not 2 <= max_candidates <= 8:
        raise ValueError("max_candidates must be between 2 and 8")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if isinstance(data_id, str):
        data_ids = tuple(value.strip() for value in data_id.split(",") if value.strip())
    else:
        data_ids = tuple(data_id)
    if not data_ids:
        raise ValueError("data_id must contain at least one dataset")
    if data_dir:
        df, boundaries, per_id = load_datasets_from_dir(data_ids, data_dir)
        lang_pairs = {value: "unknown" for value in data_ids}
    else:
        df, boundaries, lang_pairs, per_id = _load_datasets(data_ids)
    n_items = len(df)
    model, tokenizer = load_model_tokenizer(
        model_path, **kwargs.get("mt_vllm_kwargs", {})
    )
    output = func_call(
        model_path=model_path,
        src_list=df.src_text.tolist() * runs,
        src_langs=df.src_lang.tolist() * runs,
        trg_langs=df.trg_lang.tolist() * runs,
        max_candidates=max_candidates,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        retry=retry,
        protocol="simple",
        model=model,
        tokenizer=tokenizer,
    )
    predictions = output["responses"]
    expected = n_items * runs
    if len(predictions) != expected:
        raise ValueError(f"Expected {expected} predictions, got {len(predictions)}")
    _release_vllm_model(model)

    metric_results = {did: {} for did in data_ids}
    metric_none = {did: {} for did in data_ids}
    per_item = {did: {} for did in data_ids}
    valid_metrics = {did: [] for did in data_ids}
    all_metrics = []
    for metric in metrics:
        if metric == "oss":
            oss_path = kwargs.get("oss_model_path", "openai/gpt-oss-120b")
            oss_model = run_oss_SQM.init_oss_model(
                oss_path, **kwargs.get("oss_vllm_kwargs", {})
            )
            scores = run_oss_eval(df, predictions, runs, oss_model, oss_path)
            _release_vllm_model(oss_model)
        elif metric == "bleurt":
            scores = run_bleurt_eval(
                df, predictions, runs, kwargs.get("bleurt_model_path")
            )
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
        "method": "ffgqm",
        "protocol": "simple",
        "candidate_count_mean": (sum(counts) / len(counts)) if counts else 0.0,
        "candidate_count_min": min(counts) if counts else 0,
        "candidate_count_max": max(counts) if counts else 0,
        "candidate_count_distribution": {
            str(value): counts.count(value) for value in sorted(set(counts))
        },
        "parser_failures": sum(not value for value in output["parser_valid"]),
        "generation_failures": sum(
            value == "Translation Failed." for value in predictions
        ),
    }
    print(
        f"method: ffgqm | model: {model_name} | protocol: simple | "
        f"max_candidates: {max_candidates} | runs: {runs}"
    )
    print(
        f"candidate_count: mean={summary['candidate_count_mean']:.2f}, "
        f"range=[{summary['candidate_count_min']}, {summary['candidate_count_max']}]"
    )
    print(
        f"failures: parser={summary['parser_failures']}, "
        f"generation={summary['generation_failures']}"
    )
    for did in data_ids:
        print(f"\n--- {did} ---")
        for metric, value in metric_results[did].items():
            print(f"{metric}: {value:.6f} (none={metric_none[did][metric]})")

    if save_results:
        extra_items = []
        for index in range(n_items):
            extra_items.append({
                "candidates": [
                    output["candidates"][run * n_items + index]
                    for run in range(runs)
                ],
                "candidate_count": [
                    output["candidate_counts"][run * n_items + index]
                    for run in range(runs)
                ],
                "scores": [
                    output["scores"][run * n_items + index]
                    for run in range(runs)
                ],
                "selected_candidate_indices": [
                    output["selected_candidate_indices"][run * n_items + index]
                    for run in range(runs)
                ],
                "raw_output": [
                    output["raw_outputs"][run * n_items + index]
                    for run in range(runs)
                ],
                "parser_valid": [
                    output["parser_valid"][run * n_items + index]
                    for run in range(runs)
                ],
            })
        save_ffgpe_results_by_dataset(
            df=df,
            boundaries=boundaries,
            per_id=per_id,
            data_ids=data_ids,
            predictions=predictions,
            runs=runs,
            metric_results=metric_results,
            per_item_metrics=per_item,
            valid_metrics=valid_metrics,
            model_name=model_name,
            model_path=model_path,
            prompt_type="fixed_4",
            extra_items=extra_items,
            metadata={
                "method": "ffgqm",
                "protocol": "simple",
                "max_candidates": max_candidates,
                "summary": summary,
            },
        )
        payload = {
            "method": "ffgqm",
            "data_name": list(data_ids),
            "model_name": model_name,
            "model_path": model_path,
            "protocol": "simple",
            "max_candidates": max_candidates,
            "metrics": metric_results,
            "summary": summary,
            "items": [],
        }
        for index in range(n_items):
            item_did = next(did for did, (start, end) in boundaries.items() if start <= index < end)
            local_index = index - boundaries[item_did][0]
            payload["items"].append({
                "source": df.iloc[index].src_text,
                "reference": df.iloc[index].trg_text,
                "predictions": [predictions[run * n_items + index] for run in range(runs)],
                "metrics_avg": {
                    metric: per_item[item_did][metric][local_index]
                    for metric in valid_metrics[item_did]
                },
                "candidates": [output["candidates"][run * n_items + index] for run in range(runs)],
                "scores": [output["scores"][run * n_items + index] for run in range(runs)],
                "selected_candidate_indices": [output["selected_candidate_indices"][run * n_items + index] for run in range(runs)],
                "raw_output": [output["raw_outputs"][run * n_items + index] for run in range(runs)],
                "parser_valid": [output["parser_valid"][run * n_items + index] for run in range(runs)],
            })
        Path(f"{model_name}__ffgqm.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    log_results_to_wandb(
        metric_results,
        {
            "model_name": model_name,
            "model_path": model_path,
            "metrics": all_metrics,
            "method": "ffgqm",
            "protocol": "simple",
            "max_candidates": max_candidates,
        },
        metric_none,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
