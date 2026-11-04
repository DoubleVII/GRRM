"""Evaluate scored predictions against reference-assisted difficulty annotations."""
import json
from pathlib import Path

from eval.difficulty_eval_core import evaluate, grouped_summary
from utils.difficulty_protocol import load_annotations, load_predictions, save_artifact
from utils.hard_protocol import output_paths, parse_data_ids, normalize_bool, check_output


def main(data_id, predictions_dir, annotations_dir, output_dir, mode="direct", model_path=None,
         temperature=0.9, top_p=1.0, max_tokens=8192, retry=3,
         reasoning_effort="medium", vllm_kwargs=None, score_only=False, exclude_calibration=True,
         failure_mode="auto", span_backend="llm", span_proxy_options=None,
         span_prompt_variant="pairwise", span_match_policy="semantic"):
    """Use span_backend=heuristic for local overlap rewards; llm preserves semantic matching.

    span_proxy_options accepts a dict, e.g. {"unit":"token","method":"dice"}.
    The heuristic defaults were selected on source-disjoint development data.
    span_backend=inst uses a small instruct judge with native thinking disabled.
    span_match_policy=localization asks OSS to ignore translation-solution correctness.
    """
    if span_backend not in ('llm', 'heuristic', 'inst'):
        raise ValueError('span_backend must be llm, heuristic or inst')
    if failure_mode not in ('auto', 'strict', 'fallback'):
        raise ValueError('failure_mode must be auto, strict or fallback')
    data_ids = parse_data_ids(data_id)
    paths = output_paths(data_ids, output_dir, "evaluation", mode)
    summary_path = Path(output_dir) / f"{mode}.summary.json"
    check_output(summary_path)
    predictions, prediction_configs = load_predictions(data_ids, predictions_dir, mode)
    run_counts = {config.get('runs', 1) for config in prediction_configs.values()}
    if len(run_counts) != 1:
        raise ValueError('Datasets must have the same prediction runs for joint evaluation')
    runs = run_counts.pop()
    golden, golden_configs = load_annotations(data_ids, annotations_dir)
    # Honor each dataset's saved prediction and annotation span limits.
    score_only, exclude_calibration = normalize_bool(score_only), normalize_bool(exclude_calibration)
    print(f"Evaluation: datasets={len(data_ids)}; predictions={len(predictions)}; "
          f"runs={runs}; score_only={score_only}; span_backend={span_backend}", flush=True)
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                   retry=retry, reasoning_effort=reasoning_effort, vllm_kwargs=vllm_kwargs)
    def limits(configs, item):
        return {k: configs[item["data_id"]][k] for k in ("max_span_chars", "max_span_words")
                if k in configs[item["data_id"]]}
    items, summary = evaluate(predictions, golden, model_path,
        score_only=score_only, exclude_calibration=exclude_calibration,
        span_backend=span_backend, span_proxy_options=span_proxy_options,
        span_prompt_variant=span_prompt_variant,
        span_match_policy=span_match_policy,
        prediction_limits=lambda item: limits(prediction_configs, item),
        reference_limits=lambda item: limits(golden_configs, item),
        failure_mode=lambda item: (prediction_configs[item['data_id']].get('failure_mode', 'strict')
                                   if failure_mode == 'auto' else failure_mode), **options)
    for did, path in paths.items():
        selected = [item for item in items if item["data_id"] == did]
        save_artifact(path, "evaluation", dict(model_path=model_path, data_id=did, mode=mode, **options,
            score_only=score_only, exclude_calibration=exclude_calibration, failure_mode=failure_mode, runs=runs,
            span_backend=span_backend, span_proxy_options=span_proxy_options,
            span_prompt_variant=span_prompt_variant,
            span_match_policy=span_match_policy,
            prediction_config=prediction_configs[did], annotation_config=golden_configs[did]),
            selected, summary=grouped_summary(selected))
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    # Log one compact line per group; detailed per-run statistics stay in JSON.
    for label, value in [("overall (pooled)", summary['overall']), *summary['by_dataset'].items()]:
        scores, spans = value['scores'], value['spans']
        brief = dict(included=value['included'], excluded=value['excluded'],
            score_failed=scores['failed'], mae=scores['mae'], rmse=scores['rmse'],
            spearman=scores['spearman'], span_failed=spans['failed'],
            precision=spans['precision'], recall=spans['recall'], f1=spans['f1'])
        print(f"Evaluation {label}: {json.dumps(brief, ensure_ascii=False, allow_nan=False)}", flush=True)
    print(f"Evaluation summary saved: {summary_path}", flush=True)
    return summary


if __name__ == "__main__":
    import fire
    fire.Fire(main, serialize=lambda result: None)
