#!/usr/bin/env python3
"""Report a fixed difficulty-balanced preset as four rows and export TSV.

Within each difficulty, per-item evaluation results are pooled and all metrics
are recomputed, including RMSE, correlations, and span P/R/F1. Overall is the
arithmetic mean of the three difficulty metrics (1/3 each).
N always shows the actual total included record count. All prediction runs are
included. Missing metrics propagate as null instead of changing the weights.
LTB-dev and all other datasets outside the preset are ignored.

Usage:
    .venv/bin/python scripts/report_hard_eval_balanced.py path/to/direct.summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_hard_eval import DEFAULT_COLUMNS, flatten_metrics, select_datasets, write_table


GROUPS = {
    'low (WMT)': ('wmt23_zh_en', 'wmt24pp_en_zh'),
    'medium (SeedX)': ('seedx_challenge_zhen', 'seedx_challenge_enzh'),
    'high (LTB-eval)': ('ltb_v1_eval_scarcity_high', 'ltb_v1_eval_scarcity_medium',
                        'ltb_v1_eval_scarcity_low'),
}


def balanced_rows(data: dict, json_path: Path) -> list:
    """Pool evaluation items within groups, then weight group metrics equally."""
    datasets = data.get('by_dataset')
    if not isinstance(datasets, dict):
        raise ValueError('Expected a summary JSON with a by_dataset object')
    missing = [name for names in GROUPS.values() for name in names if name not in datasets]
    if missing:
        raise ValueError(f"Missing preset datasets: {', '.join(missing)}")
    rows = []
    for label, names in GROUPS.items():
        summary = select_datasets(data, json_path, names)['overall']
        if summary['included'] == 0:
            raise ValueError(f'{label}: no included evaluation records')
        values = flatten_metrics(summary)
        rows.append([label, *(values.get(path) for _, path in DEFAULT_COLUMNS)])
    overall = ['overall (1/3 each)', sum(row[1] for row in rows)]
    for index in range(2, len(DEFAULT_COLUMNS) + 1):
        metrics = [row[index] for row in rows]
        overall.append(None if any(v is None for v in metrics) else sum(metrics) / 3)
    return [*rows, overall]


def main(json_file: str, output: str | None = None, decimals: int = 4) -> None:
    """Print the fixed four-row preset and export a tab-separated file.

    Args:
        json_file: Evaluation summary JSON; requires sibling dataset evaluation
            JSON files to recompute metrics from individual records.
        output: TSV path; defaults to the input with a .balanced.tsv suffix.
        decimals: Terminal decimal places, 0 to 15; TSV retains full precision.
    """
    if type(decimals) is not int or not 0 <= decimals <= 15:
        raise fire.core.FireError('--decimals must be an integer between 0 and 15')
    source = Path(json_file)
    destination = Path(output) if output is not None else source.with_suffix('.balanced.tsv')
    if destination.resolve() == source.resolve():
        raise fire.core.FireError('TSV output must differ from the input JSON')
    try:
        data = json.loads(source.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('Expected a summary JSON object')
        rows = balanced_rows(data, source)
        write_table(['setting', *(label for label, _ in DEFAULT_COLUMNS)], rows,
                    destination, decimals)
    except (OSError, ValueError) as exc:
        raise fire.core.FireError(str(exc)) from exc


if __name__ == '__main__':
    fire.Fire(main)
