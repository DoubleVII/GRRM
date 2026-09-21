from typing import Iterable, Optional

from eval.run_mt_eval import (
    _load_datasets,
    _release_vllm_model,
    log_results_to_wandb,
    run_bleurt_eval,
    run_oss_eval,
    run_oss_SQM,
    save_results_to_json,
)
from inference.run_inst_flash_gqm_mt import init_inst_model, run_pipeline
from inference.run_oss_diverse_mt import normalize_bool
from utils.helpers import build_notes_list, load_datasets_from_dir, split_metrics_by_notes


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
    candidate_count: int = 4,
    candidate_temperature: float = 1.0,
    candidate_top_p: float = 1.0,
    candidate_top_k: int = 0,
    candidate_presence_penalty: float = 0.0,
    candidate_repetition_penalty: float = 1.0,
    candidate_max_new_tokens: int = 8192,
    gqm_temperature: float = 1.0,
    gqm_top_p: float = 1.0,
    gqm_top_k: int = 0,
    gqm_presence_penalty: float = 0.0,
    gqm_repetition_penalty: float = 1.0,
    gqm_max_new_tokens: int = 8192,
    metrics: list[str] = ["bleurt", "oss"],
    runs: int = 1,
    save_results: bool = False,
    data_dir: Optional[str] = None,
    difficulty_filter: int = 0,
    retry: int = 3,
    enable_thinking: bool = True,
    **kwargs,
):
    """Evaluate instruct FlashGQM with the standard MT metrics."""
    if not 2 <= candidate_count <= 8:
        raise ValueError("candidate_count must be between 2 and 8")
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
    flat_notes, notes_mask = build_notes_list(frame, difficulty_filter, runs)
    flat_size = item_count * runs

    engine = init_inst_model(model_path, **kwargs.get("mt_vllm_kwargs", {}))
    pipeline = run_pipeline(
        frame["src_text"].tolist() * runs,
        frame["src_lang"].tolist() * runs,
        frame["trg_lang"].tolist() * runs,
        model=engine,
        max_candidates=candidate_count,
        candidate_counts=[candidate_count] * flat_size,
        candidate_temperature=candidate_temperature,
        candidate_top_p=candidate_top_p,
        candidate_top_k=candidate_top_k,
        candidate_presence_penalty=candidate_presence_penalty,
        candidate_repetition_penalty=candidate_repetition_penalty,
        candidate_max_tokens=candidate_max_new_tokens,
        gqm_temperature=gqm_temperature,
        gqm_top_p=gqm_top_p,
        gqm_top_k=gqm_top_k,
        gqm_presence_penalty=gqm_presence_penalty,
        gqm_repetition_penalty=gqm_repetition_penalty,
        gqm_max_tokens=gqm_max_new_tokens,
        retry=retry,
        enable_thinking=enable_thinking,
    )
    raw_predictions = pipeline["gqm"]["translations"]
    predictions = [value or "Translation Failed." for value in raw_predictions]
    if len(predictions) != flat_size:
        raise ValueError(f"Expected {flat_size} predictions, got {len(predictions)}")
    _release_vllm_model(engine.model)
    del engine

    metric_results = {value: {} for value in data_ids}
    metric_none = {value: {} for value in data_ids}
    per_item_metrics = {value: {} for value in data_ids}
    valid_metrics = {value: [] for value in data_ids}
    metrics_by_notes = {value: {} for value in data_ids}
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
        else:
            scores = run_bleurt_eval(
                frame, predictions, runs, kwargs.get("bleurt_model_path")
            )
        split = split_metrics_by_notes(
            scores, notes_mask, boundaries, item_count, runs
        )
        for current_id in data_ids:
            metric_results[current_id][metric] = split[current_id]["all"]["avg"]
            metric_none[current_id][metric] = split[current_id]["all"]["none_count"]
            per_item_metrics[current_id][metric] = split[current_id]["all"][
                "per_item_avgs"
            ]
            metrics_by_notes[current_id][metric] = split[current_id]
            valid_metrics[current_id].append(metric)
        evaluated_metrics.append(metric)

    candidate_failures = sum(
        count < 2 for count in pipeline["usable_candidate_counts"]
    )
    gqm_failures = sum(
        count >= 2 and value is None
        for count, value in zip(
            pipeline["usable_candidate_counts"], raw_predictions
        )
    )
    print(
        f"method=inst_flash_gqm | model={model_name} | "
        f"candidate_count={candidate_count} | runs={runs}"
    )
    print(
        f"candidate_generation_failures={candidate_failures} | "
        f"gqm_failures={gqm_failures}"
    )
    for current_id in data_ids:
        print(f"\n=== {current_id} ===")
        for metric, value in metric_results[current_id].items():
            print(f"{metric}: {value:.6f} (none={metric_none[current_id][metric]})")

    if save_results:
        for current_id, (start, end) in boundaries.items():
            size = end - start
            nested = [
                [
                    predictions[run * item_count + start + index]
                    for index in range(size)
                ]
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
                temperature=gqm_temperature,
                top_p=gqm_top_p,
                max_new_tokens=gqm_max_new_tokens,
                runs=runs,
                prompt_type=f"inst-flash-gqm-n{candidate_count}",
                difficulty_filter=difficulty_filter,
                notes_list_per_item=flat_notes[start:end],
                use_notes_mask_per_item=notes_mask[start:end],
            )

    log_results_to_wandb(
        metric_results,
        {
            "dataset_names": data_ids,
            "model_path": model_path,
            "model_name": model_name,
            "method": "inst_flash_gqm",
            "candidate_count": candidate_count,
            "candidate_temperature": candidate_temperature,
            "candidate_top_p": candidate_top_p,
            "candidate_top_k": candidate_top_k,
            "candidate_presence_penalty": candidate_presence_penalty,
            "candidate_repetition_penalty": candidate_repetition_penalty,
            "candidate_max_new_tokens": candidate_max_new_tokens,
            "gqm_temperature": gqm_temperature,
            "gqm_top_p": gqm_top_p,
            "gqm_top_k": gqm_top_k,
            "gqm_presence_penalty": gqm_presence_penalty,
            "gqm_repetition_penalty": gqm_repetition_penalty,
            "gqm_max_new_tokens": gqm_max_new_tokens,
            "tie_break": "first_maximum",
            "runs": runs,
            "metrics": evaluated_metrics,
            "lang_pairs": lang_pairs,
            "prompt_type": "markdown",
            "difficulty_filter": difficulty_filter,
            "data_dir": data_dir,
            "enable_thinking": enable_thinking,
            "retry": retry,
        },
        metric_none,
        metrics_by_notes,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
