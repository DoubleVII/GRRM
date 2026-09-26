"""Evaluate translations already present in the LTB v1 JSON.

The metric implementation is shared with ``eval.run_mt_eval``. This script
does not run MT inference: ``--model_name`` selects a translation entry from
the original LTB JSON, while BLEURT and gpt-oss score that entry against the
prepared LTB reference and comment columns.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import fire
import pandas as pd


from eval.run_mt_eval import _release_vllm_model, run_bleurt_eval, run_oss_eval
from utils.config import MT_TEST_DATA_META_INFO


DEFAULT_INPUT_PATH = "/home/nfs06/yangs/data/hf/zouhar/last-translation-benchmark/data/v1.json"
DEFAULT_OSS_MODEL_PATH = "openai/gpt-oss-120b"


def _nonempty(value) -> bool:
    return value is not None and (
        not isinstance(value, str) or bool(value.strip())
    )


def _parse_metrics(metrics) -> list[str]:
    if isinstance(metrics, str):
        metrics = [item.strip() for item in metrics.split(",") if item.strip()]
    metrics = list(metrics)
    unsupported = set(metrics) - {"bleurt", "oss"}
    if unsupported or not metrics:
        raise ValueError(
            f"metrics must contain only 'bleurt' and/or 'oss'; got {metrics}"
        )
    return metrics


def _load_raw(input_path: str) -> list[dict]:
    with open(input_path, encoding="utf-8") as source_file:
        rows = json.load(source_file)
    if not isinstance(rows, list):
        raise ValueError("LTB input must be a JSON list")
    return rows


def model_coverage(rows: list[dict]) -> dict[str, dict]:
    """Return output coverage for every model, including empty translations."""
    all_ids = {row["id"] for row in rows}
    by_model: dict[str, set[int]] = defaultdict(set)
    nonempty_by_model: dict[str, set[int]] = defaultdict(set)
    empty_by_model: dict[str, int] = defaultdict(int)
    duplicate_by_model: dict[str, int] = defaultdict(int)
    for row in rows:
        seen_models: set[str] = set()
        for translation in row.get("translations", []):
            model = translation.get("model")
            if model is None:
                continue
            if model in seen_models:
                duplicate_by_model[model] += 1
                continue
            seen_models.add(model)
            by_model[model].add(row["id"])
            if _nonempty(translation.get("translation")):
                nonempty_by_model[model].add(row["id"])
            else:
                empty_by_model[model] += 1

    result = {}
    for model in sorted(by_model):
        ids = by_model[model]
        nonempty_ids = nonempty_by_model[model]
        result[model] = {
            "rows": len(ids),
            "coverage": len(ids) / len(all_ids) if all_ids else 0.0,
            "nonempty_rows": len(nonempty_ids),
            "nonempty_coverage": len(nonempty_ids) / len(all_ids) if all_ids else 0.0,
            "empty_rows": empty_by_model[model],
            "missing_rows": len(all_ids - ids),
            "missing_nonempty_rows": len(all_ids - nonempty_ids),
            "duplicate_rows": duplicate_by_model[model],
        }
    return result


def _translation_map(rows: list[dict], model_name: str) -> dict[int, str]:
    output = {}
    for row in rows:
        matches = [
            item for item in row.get("translations", [])
            if item.get("model") == model_name
        ]
        if len(matches) > 1:
            raise ValueError(f"LTB id {row['id']}: duplicate output for {model_name!r}")
        if matches and _nonempty(matches[0].get("translation")):
            output[row["id"]] = matches[0]["translation"]
    return output


def _parse_data_ids(data_id) -> tuple[str, ...]:
    if data_id is None:
        raise ValueError("data_id is required; pass one or more configured data IDs")
    if isinstance(data_id, str):
        values = tuple(item.strip() for item in data_id.split(",") if item.strip())
    else:
        values = tuple(data_id)
    if not values:
        raise ValueError("data_id must contain at least one dataset identifier")
    unknown = [item for item in values if item not in MT_TEST_DATA_META_INFO]
    if unknown:
        raise ValueError(f"Unknown data_id(s): {unknown}")
    return values


def _paths_for_data_ids(data_ids: tuple[str, ...]) -> dict[str, Path]:
    """Resolve paths exactly as the standard MT evaluator does."""
    paths = {}
    for data_id in data_ids:
        configured = Path(MT_TEST_DATA_META_INFO[data_id]["path"])
        paths[data_id] = configured if configured.is_absolute() else REPO_ROOT / configured
    return paths


def build_evaluation_frames(
    raw_rows: list[dict],
    model_name: str,
    data_id=None,
    # Deprecated test-only compatibility. Dataset selection remains data_id-based.
    data_root: Optional[str] = None,
    subset: Optional[str] = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    prediction_by_id = _translation_map(raw_rows, model_name)
    frames = {}
    coverage = {}
    data_ids = _parse_data_ids(data_id)
    paths = _paths_for_data_ids(data_ids)
    # Allow legacy tests to redirect configured paths under a temporary root.
    if data_root:
        root = Path(data_root)
        paths = {
            did: root / next(
                part for part in Path(path).parts if part.startswith("ltb-v1-")
            ) / Path(path).name
            for did, path in paths.items()
        }
    for current_data_id, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Prepared LTB dataset not found: {path}")
        frame = pd.read_parquet(path)
        available = frame["ltb_id"].isin(prediction_by_id)
        evaluated = frame.loc[available].copy().reset_index(drop=True)
        evaluated["mt_text"] = evaluated["ltb_id"].map(prediction_by_id)
        frames[current_data_id] = evaluated
        coverage[current_data_id] = {
            "total_rows": len(frame),
            "evaluated_rows": len(evaluated),
            "missing_rows": int((~available).sum()),
            "coverage": len(evaluated) / len(frame) if len(frame) else 0.0,
        }
    return frames, coverage


def _score_frames(
    frames: dict[str, pd.DataFrame],
    metrics: list[str],
    bleurt_model_path: Optional[str],
    oss_model_path: str,
    oss_vllm_kwargs: Optional[dict] = None,
) -> dict[str, dict]:
    nonempty = [(data_id, frame) for data_id, frame in frames.items() if len(frame)]
    if not nonempty:
        raise ValueError("The selected model has no non-empty translations in the selected data_id(s)")
    combined = pd.concat([frame for _, frame in nonempty], ignore_index=True)
    boundaries = {}
    offset = 0
    for data_id, frame in nonempty:
        boundaries[data_id] = (offset, offset + len(frame))
        offset += len(frame)
    scores: dict[str, list] = {}
    if "oss" in metrics:
        import inference.run_oss_SQM as run_oss_SQM

        oss_model = run_oss_SQM.init_oss_model(
            oss_model_path, **(oss_vllm_kwargs or {})
        )
        try:
            scores["oss"] = run_oss_eval(
                combined,
                combined["mt_text"].tolist(),
                1,
                oss_model,
                oss_model_path=oss_model_path,
            )
        finally:
            _release_vllm_model(oss_model)
    if "bleurt" in metrics:
        scores["bleurt"] = run_bleurt_eval(
            combined,
            combined["mt_text"].tolist(),
            1,
            bleurt_model_path=bleurt_model_path,
        )

    results = {data_id: {} for data_id, _ in nonempty}
    for metric, values in scores.items():
        for data_id, (start, end) in boundaries.items():
            values_for_dataset = [value for value in values[start:end] if value is not None]
            results[data_id][metric] = (
                sum(values_for_dataset) / len(values_for_dataset)
                if values_for_dataset
                else None
            )
            results[data_id][f"{metric}_none_count"] = end - start - len(values_for_dataset)
    return results


def main(
    model_name: Optional[str] = None,
    data_id=None,
    input_path: str = DEFAULT_INPUT_PATH,
    metrics: list[str] = ("bleurt", "oss"),
    bleurt_model_path: Optional[str] = None,
    oss_model_path: str = None,
    oss_vllm_kwargs: Optional[dict] = None,
    require_complete: bool = False,
    coverage_only: bool = False,
    output_path: Optional[str] = None,
):
    metrics = _parse_metrics(metrics)
    raw_rows = _load_raw(input_path)
    selected_data_ids = _parse_data_ids(data_id)
    selected_paths = _paths_for_data_ids(selected_data_ids)
    selected_ids = set()
    for path in selected_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Prepared LTB dataset not found: {path}")
        selected_ids.update(pd.read_parquet(path, columns=["ltb_id"])["ltb_id"].tolist())
    selected_raw_rows = [row for row in raw_rows if row["id"] in selected_ids]
    coverage = model_coverage(selected_raw_rows)
    if coverage_only:
        payload = {
            "data_id": selected_data_ids,
            "coverage_scope_rows": len(selected_ids),
            "coverage_all_models": coverage,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if output_path:
            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(payload, output_file, ensure_ascii=False, indent=2)
            print(f"Saved results to {output_path}")
        return None
    if model_name not in coverage:
        raise ValueError(f"Unknown model {model_name!r}; available models: {sorted(coverage)}")

    frames, data_id_coverage = build_evaluation_frames(
        raw_rows, model_name, data_id=selected_data_ids
    )
    if require_complete:
        incomplete = {
            data_id: info for data_id, info in data_id_coverage.items()
            if info["missing_rows"]
        }
        if incomplete:
            raise ValueError(f"Model {model_name!r} is incomplete: {incomplete}")

    payload = {
        "model_name": model_name,
        "data_id": selected_data_ids,
        "metrics": metrics,
        "coverage_all_models": coverage,
        "coverage": coverage[model_name],
        "data_id_coverage": data_id_coverage,
    }
    if not coverage_only:
        payload["results"] = _score_frames(
            frames, metrics, bleurt_model_path, oss_model_path, oss_vllm_kwargs
        )
        for data_id, values in payload["results"].items():
            print(f"\n=== {data_id} ===")
            for metric in metrics:
                print(f"  {metric}: {values.get(metric)} (none_count={values.get(metric + '_none_count', 0)})")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if output_path:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
        print(f"Saved results to {output_path}")
    return None


if __name__ == "__main__":
    fire.Fire(main)
