#!/usr/bin/env python3
"""Print a difficulty evaluation summary as a table and export it to TSV.

Usage:
    .venv/bin/python scripts/report_hard_eval.py path/to/direct.summary.json
    .venv/bin/python scripts/report_hard_eval.py summary.json --output table.tsv

Rows follow by_dataset order, with overall last. Repeated predictions use the
top-level pooled summary, matching the evaluation log. Rates stay on the 0–1
scale. Terminal values are rounded; TSV retains the JSON's numeric precision.
Use --all-metrics to include every numeric summary field, including counts and
coverage. Null values appear as '-' in the terminal and empty cells in TSV.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import fire


DEFAULT_COLUMNS = (
    ("N", "included"),
    ("MAE", "scores.mae"),
    ("RMSE", "scores.rmse"),
    ("Bias", "scores.bias"),
    ("ExactAcc", "scores.exact_accuracy"),
    ("Within1Acc", "scores.within_one_accuracy"),
    ("Spearman", "scores.spearman"),
    ("KendallTauB", "scores.kendall_tau_b"),
    ("SpanP", "spans.precision"),
    ("SpanR", "spans.recall"),
    ("SpanF1", "spans.f1"),
)


def flatten_metrics(summary: dict, prefix: str = "") -> dict:
    """Flatten numeric and null fields while retaining unambiguous paths."""
    result = {}
    for key, value in summary.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_metrics(value, name))
        elif type(value) in (int, float) or value is None:
            result[name] = value
    return result


def main(json_file: str, output: str | None = None, decimals: int = 4,
         all_metrics: bool = False) -> None:
    """Print dataset/overall metrics and export a tab-separated file.

    Args:
        json_file: Evaluation summary JSON.
        output: TSV path; defaults to the input with a .tsv suffix.
        decimals: Terminal decimal places, from 0 to 15 (TSV keeps full precision).
        all_metrics: Include all numeric summary fields.
    """
    if type(decimals) is not int or not 0 <= decimals <= 15:
        raise fire.core.FireError("--decimals must be an integer between 0 and 15")
    json_path = Path(json_file)
    output = Path(output) if output is not None else json_path.with_suffix(".tsv")
    if output.resolve() == json_path.resolve():
        raise fire.core.FireError("TSV output must differ from the input JSON")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or not isinstance(data.get("overall"), dict)
                or not isinstance(data.get("by_dataset"), dict)
                or any(not isinstance(value, dict) for value in data["by_dataset"].values())):
            raise ValueError("Expected a summary JSON with overall and by_dataset objects")
        groups = [*data["by_dataset"].items(), ("overall", data["overall"])]
        flattened = [(name, flatten_metrics(value)) for name, value in groups]
        columns = DEFAULT_COLUMNS
        if all_metrics:
            # Exclude null text fields (e.g. correlation_reason) while retaining
            # metrics that are undefined for every dataset, such as correlation.
            paths = dict.fromkeys(key for _, values in flattened for key in values
                                  if key != "scores.correlation_reason")
            columns = [(key, key) for key in paths]
        headers = ["dataset", *(label for label, _ in columns)]
        rows = [[name, *(values.get(path) for _, path in columns)]
                for name, values in flattened]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(headers)
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        raise fire.core.FireError(str(exc)) from exc

    def display(value):
        if value is None:
            return "-"
        return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)

    formatted = [[display(value) for value in row] for row in rows]
    widths = [max(len(row[i]) for row in [headers, *formatted])
              for i in range(len(headers))]

    def print_row(row):
        print(" | ".join(value.ljust(width) if i == 0 else value.rjust(width)
                         for i, (value, width) in enumerate(zip(row, widths))))

    print_row(headers)
    print("-+-".join("-" * width for width in widths))
    for row in formatted:
        print_row(row)
    print(f"TSV: {output}", file=sys.stderr)


if __name__ == "__main__":
    fire.Fire(main)
