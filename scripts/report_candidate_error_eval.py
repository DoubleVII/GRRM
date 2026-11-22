"""Recompute candidate-error dataset or difficulty-balanced tables from saved rows."""
import json
from pathlib import Path

from eval.run_candidate_error_eval import report_rows, _names
from scripts.report_hard_eval import write_table
from scripts.report_hard_eval_balanced import GROUPS
from utils.candidate_error_protocol import digest
from utils.jsonl import read_jsonl


COLUMNS = ('N', 'Precision', 'Recall', 'F1', 'AP', 'SpanSpearman', 'Triggers/100')


def metrics(rows, scoring, threshold, mapping):
    report = report_rows(rows, scoring, threshold, mapping, n_bootstrap=0)
    return [len(rows), *(report['overall'][k] for k in ('precision', 'recall', 'f1')),
            report['ap'], report['reliability']['span_correctness_spearman'],
            report['overall']['triggers_per_100']]


def table_rows(report_file, data_ids=None, split='test', cohort='natural', balanced=False):
    if isinstance(balanced, str) and balanced.lower() in ('true', 'false'):
        balanced = balanced.lower() == 'true'
    if not isinstance(balanced, bool):
        raise ValueError('balanced must be boolean')
    source = Path(report_file)
    report = json.loads(source.read_text(encoding='utf-8'))
    saved = read_jsonl(source.parent/'evaluation_rows.jsonl')
    if digest(saved) != report['evaluation_rows_digest']:
        raise ValueError('Evaluation rows do not match report.json')
    rows = [r for r in saved if not r['evaluation_excluded_reason'] and
            r['item']['split'] == split and r['item']['cohort'] == cohort]
    names = _names(data_ids, 'data_ids')
    available = {r['item']['data_id'] for r in rows}
    if names and set(names) - available:
        raise ValueError(f'Unknown or empty datasets in selected split/cohort: {sorted(set(names)-available)}')
    if names:
        rows = [r for r in rows if r['item']['data_id'] in names]
    if not rows:
        raise ValueError('No included evaluation records selected')
    config = {key: report[key] for key in ('scoring', 'threshold', 'mapping')}
    if balanced:
        if names:
            raise ValueError('balanced and data_ids are mutually exclusive')
        missing = {name for group in GROUPS.values() for name in group} - available
        if missing:
            raise ValueError(f'Missing balanced preset datasets: {sorted(missing)}')
        table = [[label, *metrics([r for r in rows if r['item']['data_id'] in group], **config)]
                 for label, group in GROUPS.items()]
        overall = ['overall (1/3 each)', sum(row[1] for row in table)]
        for i in range(2, len(COLUMNS)+1):
            values = [row[i] for row in table]
            overall.append(None if any(v is None for v in values) else sum(values)/3)
        return [*table, overall]
    names = names or sorted(available)
    return [[name, *metrics([r for r in rows if r['item']['data_id'] == name], **config)]
            for name in names] + [['overall', *metrics(rows, **config)]]


def main(report_file, output=None, data_ids=None, split='test', cohort='natural',
         balanced=False, decimals=4):
    """Read sibling evaluation_rows.jsonl; no matching or threshold fitting."""
    if type(decimals) is not int or not 0 <= decimals <= 15:
        raise ValueError('decimals must be an integer between 0 and 15')
    if isinstance(balanced, str) and balanced.lower() in ('true', 'false'):
        balanced = balanced.lower() == 'true'
    source = Path(report_file)
    suffix = '.balanced.tsv' if balanced else '.selected.tsv' if data_ids else '.tsv'
    destination = Path(output) if output else source.with_suffix(suffix)
    if destination.resolve() in (source.resolve(), (source.parent/'evaluation_rows.jsonl').resolve()):
        raise ValueError('Output must differ from evaluation input artifacts')
    table = table_rows(source, data_ids, split, cohort, balanced)
    write_table(['dataset', *COLUMNS], table, destination, decimals)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
