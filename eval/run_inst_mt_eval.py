from typing import Iterable, Optional

from eval.run_mt_eval import (
    _load_datasets,
    _release_vllm_model,
    _split_scores_by_data_id,
    log_results_to_wandb,
    run_bleurt_eval,
    run_oss_eval,
    run_oss_SQM,
    save_results_to_json,
)
from inference.run_inst_mt import init_inst_model, run_translation_stage
from inference.run_oss_diverse_mt import normalize_bool
from utils.helpers import load_datasets_from_dir


def _parse_data_ids(data_id) -> tuple[str, ...]:
    if isinstance(data_id, str):
        values = tuple(value.strip() for value in data_id.split(",") if value.strip())
    elif isinstance(data_id, Iterable):
        values = tuple(data_id)
    else:
        values = (data_id,)
    if not values:
        raise ValueError("data_id must contain at least one dataset")
    return values


def main(
    data_id: tuple[str],
    model_path: str,
    model_name: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 8192,
    metrics: list[str] = ["bleurt", "oss"],
    runs: int = 1,
    save_results: bool = False,
    data_dir: Optional[str] = None,
    retry: int = 3,
    enable_thinking: bool = False,
    prompt_version: str = "codeblock",
    **kwargs,
):
    """Evaluate ordinary instruct translation with the standard MT metrics."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    enable_thinking = normalize_bool(enable_thinking, "enable_thinking")
    data_ids = _parse_data_ids(data_id)
    if data_dir:
        frame, boundaries, per_id = load_datasets_from_dir(data_ids, data_dir)
        lang_pairs = {value: "unknown" for value in data_ids}
    else:
        frame, boundaries, lang_pairs, per_id = _load_datasets(data_ids)
    item_count = len(frame)

    engine = init_inst_model(model_path, **kwargs.get("mt_vllm_kwargs", {}))
    inference = run_translation_stage(
        frame["src_text"].tolist() * runs,
        frame["src_lang"].tolist() * runs,
        frame["trg_lang"].tolist() * runs,
        model=engine,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens,
        retry=retry,
        enable_thinking=enable_thinking,
        prompt_version=prompt_version,
    )
    raw_predictions = inference["translations"]
    predictions = [value or "Translation Failed." for value in raw_predictions]
    expected = item_count * runs
    if len(predictions) != expected:
        raise ValueError(f"Expected {expected} predictions, got {len(predictions)}")
    _release_vllm_model(engine.model)
    del engine

    metric_results = {value: {} for value in data_ids}
    metric_none = {value: {} for value in data_ids}
    per_item_metrics = {value: {} for value in data_ids}
    valid_metrics = {value: [] for value in data_ids}
    evaluated_metrics = []
    unsupported = set(metrics) - {"bleurt", "oss"}
    if unsupported:
        raise ValueError(f"Unsupported metrics: {sorted(unsupported)}")
    for metric in ("oss", "bleurt"):
        if metric not in metrics:
            continue
        if metric == "oss":
            oss_path = kwargs.get("oss_model_path", "openai/gpt-oss-120b")
            oss_model = run_oss_SQM.init_oss_model(
                oss_path, **kwargs.get("oss_vllm_kwargs", {})
            )
            scores = run_oss_eval(frame, predictions, runs, oss_model, oss_path)
            _release_vllm_model(oss_model)
        elif metric == "bleurt":
            scores = run_bleurt_eval(
                frame, predictions, runs, kwargs.get("bleurt_model_path")
            )
        split = _split_scores_by_data_id(scores, boundaries, item_count, runs)
        for current_id in data_ids:
            metric_results[current_id][metric] = split[current_id]["avg"]
            metric_none[current_id][metric] = split[current_id]["none_count"]
            per_item_metrics[current_id][metric] = split[current_id]["per_item_avgs"]
            valid_metrics[current_id].append(metric)
        evaluated_metrics.append(metric)

    print(f"method=inst_mt | model={model_name} | runs={runs}")
    print(f"generation_failures={sum(value is None for value in raw_predictions)}")
    for current_id in data_ids:
        print(f"\n=== {current_id} ===")
        for metric, value in metric_results[current_id].items():
            print(f"{metric}: {value:.6f} (none={metric_none[current_id][metric]})")

    if save_results:
        for current_id, (start, end) in boundaries.items():
            size = end - start
            nested = [
                [predictions[run * item_count + start + index] for index in range(size)]
                for run in range(runs)
            ]
            save_results_to_json(
                df=per_id[current_id],
                mt_list_for_runs_nested=nested,
                per_item_metric_avgs=per_item_metrics[current_id],
                valid_metrics=valid_metrics[current_id],
                dataset_name=current_id,
                model_name=model_name,
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                runs=runs,
                prompt_type="inst-step-by-step",
            )

    log_results_to_wandb(
        metric_results,
        {
            "dataset_names": data_ids,
            "model_path": model_path,
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "runs": runs,
            "metrics": evaluated_metrics,
            "lang_pairs": lang_pairs,
            "prompt_type": "inst-step-by-step",
            "prompt_version": prompt_version,
            "data_dir": data_dir,
            "enable_thinking": enable_thinking,
            "retry": retry,
        },
        metric_none,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
