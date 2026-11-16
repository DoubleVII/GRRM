"""Offline candidate error metrics from saved location matching records."""
import json
import math
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

from eval.candidate_error_core import (aggregate, average_precision, best_threshold, bootstrap,
    category_mapping, curve, location_edges, reliability, severity_confusion, count_item)
from utils.candidate_error_protocol import read_jsonl, write_json, write_jsonl, digest, source_cluster
from utils.span_matching import validate_pairs


def grouped(rows, threshold, scoring, mapping=None):
    result = {}
    for field in ('data_id','candidate_model','family'):
        groups = defaultdict(list)
        for r in rows:
            groups[r['item'][field]].append(r)
        result['by_'+field] = {k:aggregate(v,threshold,scoring,mapping) for k,v in groups.items()}
    values = list(result['by_data_id'].values())
    result['dataset_equal_weight'] = {m:sum(v[m] for v in values if v[m] is not None)/sum(v[m] is not None for v in values)
        if any(v[m] is not None for v in values) else None for m in ('precision','recall','f1')}
    return result


def report_rows(rows, scoring, threshold, mapping=None, n_bootstrap=2000):
    points=curve(rows,scoring,mapping)
    unique=[]
    seen=set()
    for r in rows:
        key=(r['item']['cluster'],r['item']['candidate_model'],r['item']['candidate'])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    diagnostics={}
    for mode in ('exact','overlap'):
        alternate=[dict(r,edges=location_edges(r['predicted'],r['gold'],mode)) for r in rows]
        diagnostics[mode]=aggregate(alternate,threshold,scoring,mapping)
    working = aggregate(rows,threshold,scoring,mapping)
    return dict(overall=working,
        pure_additions=dict(gold=sum(s.get('source_start') is None for r in rows for s in r['gold']),
                            predicted=sum(s.get('source_start') is None for r in rows for s in r['predicted'])),
        all_severities=aggregate(rows,threshold,scoring,mapping,major=False),
        all_severities_unfiltered=aggregate(rows,-1.,scoring,mapping,major=False),
        severity_confusion=severity_confusion(rows),curve=points,ap=average_precision(points),
        reliability=reliability(rows,scoring,mapping),
        bootstrap_ci=bootstrap(rows,threshold,scoring,mapping,n=n_bootstrap),
        deduplicated=aggregate(unique,threshold,scoring,mapping),
        location_diagnostics=diagnostics,**grouped(rows,threshold,scoring,mapping))


def _names(value, label):
    if value is None:
        return None
    values = value.split(',') if isinstance(value, str) else list(value)
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f'{label} must contain nonempty names')
    return list(dict.fromkeys(v.strip() for v in values))


def load_matching(matching_input, data_ids=None):
    """Load one matching JSONL or a directory containing dataset matching JSONLs."""
    path = Path(matching_input)
    names = _names(data_ids, 'data_ids')
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        # Only the data-layer matching artifacts, never evaluation_rows.jsonl.
        paths = sorted(path.rglob('matching.jsonl'))
    else:
        raise ValueError(f'Missing matching_input: {path}')
    rows, available = [], set()
    for file in paths:
        values = read_jsonl(file)
        available.update(r['item']['data_id'] for r in values)
        rows.extend(r for r in values if names is None or r['item']['data_id'] in names)
    if names and set(names) - available:
        raise ValueError(f'Unknown data_ids: {sorted(set(names) - available)}')
    if not rows:
        raise ValueError('No matching records selected')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate matching identities; evaluate one candidate group/variant at a time')
    modes = {r.get('match_mode') for r in rows}
    if len(modes) != 1 or not modes <= {'semantic', 'exact', 'overlap'}:
        raise ValueError('Matching records must have one consistent match_mode')
    from utils.candidate_error_protocol import identity
    for r in rows:
        if r.get('schema') != 'candidate-error-matching-v1' or r['id'] != identity(r['item']):
            raise ValueError('Invalid matching schema or candidate identity')
        if r.get('gold_status') not in ('ok', 'failed') or r.get('prediction_status') not in ('ok', 'failed'):
            raise ValueError('Invalid upstream status')
        if r.get('matching_status') not in ('ok', 'failed', 'skipped'):
            raise ValueError('Invalid matching status')
        if bool(r.get('excluded_reason')) != (r['gold_status'] != 'ok'):
            raise ValueError('Gold exclusion/status mismatch')
        if not isinstance(r.get('predicted'), list) or not isinstance(r.get('gold'), list):
            raise ValueError('Missing predicted/gold spans')
        if validate_pairs(dict(analysis='', pairs=r.get('edges')), len(r['predicted']), len(r['gold'])) is None:
            raise ValueError('Invalid matching edges')
        if r['matching_status'] != 'ok' and r['edges']:
            raise ValueError('Failed/skipped matching must have no edges')
        if r['prediction_status'] != 'ok' and r['predicted']:
            raise ValueError('Failed predictions must have no proposals')
    return rows, modes.pop()


def assign_splits(rows, *, calibration_data_ids=None, test_data_ids=None, split_file=None):
    """Unlisted records are excluded; no random partitioning or input mutation."""
    rows = deepcopy(rows)
    cal = _names(calibration_data_ids, 'calibration_data_ids')
    test = _names(test_data_ids, 'test_data_ids')
    if split_file and (cal is not None or test is not None):
        raise ValueError('split_file and dataset split lists are mutually exclusive')
    assignments = None
    if split_file:
        manifest = json.loads(Path(split_file).read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or set(manifest) - {'calibration', 'test'}:
            raise ValueError('split_file must be an object with calibration/test ID lists')
        assignments = {}
        known = {r['id'] for r in rows}
        for split, keys in manifest.items():
            if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys):
                raise ValueError('split_file values must be lists of candidate IDs')
            for key in keys:
                if key not in known:
                    raise ValueError(f'Unknown split candidate ID: {key}')
                if key in assignments:
                    raise ValueError(f'Duplicate split candidate ID: {key}')
                assignments[key] = split
    elif cal is not None or test is not None:
        cal, test = set(cal or []), set(test or [])
        if cal & test:
            raise ValueError(f'Datasets assigned to both splits: {sorted(cal & test)}')
        unknown = (cal | test) - {r['item']['data_id'] for r in rows}
        if unknown:
            raise ValueError(f'Unknown split datasets: {sorted(unknown)}')
    for r in rows:
        item = r['item']
        if assignments is not None:
            split = assignments.get(r['id'], 'excluded')
        elif cal is not None or test is not None:
            split = 'calibration' if item['data_id'] in cal else 'test' if item['data_id'] in test else 'excluded'
        else:
            split = 'test'
        item['split'] = split
        item.setdefault('cohort', 'natural')
        if item['cohort'] not in ('natural', 'contrast'):
            raise ValueError('Unknown cohort')
        item.setdefault('family', item['data_id'])
        item.setdefault('cluster', source_cluster(item))
    return rows


def _partition_keys(rows):
    return dict(ids=sorted({r['id'] for r in rows}),
                sources=sorted({source_cluster(r['item']) for r in rows}),
                ltb_ids=sorted({str(r['item']['ltb_id']) for r in rows
                                if r['item'].get('ltb_id') is not None}))


def _check_disjoint(calibration_keys, test_rows):
    test_keys = _partition_keys(test_rows)
    overlap = {key: sorted(set(calibration_keys.get(key, [])) & set(test_keys[key]))
               for key in ('ids', 'sources', 'ltb_ids')}
    if any(overlap.values()):
        affected = [r['id'] for r in test_rows if r['id'] in overlap['ids'] or
                    source_cluster(r['item']) in overlap['sources'] or
                    (r['item'].get('ltb_id') is not None and str(r['item']['ltb_id']) in overlap['ltb_ids'])]
        raise ValueError(f'Calibration/test overlap: {overlap}; test candidate IDs: {affected}')


def evaluate(matching_input, output_dir, *, scoring='probability', data_ids=None,
             calibration_data_ids=None, test_data_ids=None, split_file=None,
             calibration_file=None, threshold=None, n_bootstrap=2000):
    """Compute offline metrics. No model is imported or called.

    Dataset lists or split_file={"calibration": [candidate IDs], "test": [...]} assign
    splits explicitly; unlisted records are excluded. Without split arguments all
    selected records become test. cohort defaults to natural. A threshold is either
    explicit, selected by calibration major F1, or loaded from calibration_file.
    """
    if scoring not in ('probability', 'severity', 'categorical'):
        raise ValueError('Unknown scoring')
    if type(n_bootstrap) is not int or n_bootstrap < 0:
        raise ValueError('n_bootstrap must be a nonnegative integer')
    if calibration_file and threshold is not None:
        raise ValueError('calibration_file and threshold are mutually exclusive')
    rows, match_mode = load_matching(matching_input, data_ids)
    matching_digest = digest(rows)
    rows = assign_splits(rows, calibration_data_ids=calibration_data_ids,
                         test_data_ids=test_data_ids, split_file=split_file)
    assigned_cal = [r for r in rows if r['item']['split'] == 'calibration']
    test = [r for r in rows if r['item']['split'] == 'test']
    if not assigned_cal and not test:
        raise ValueError('No calibration or test records assigned')
    _check_disjoint(_partition_keys(assigned_cal), test)
    included = [r for r in rows if r['item']['split'] != 'excluded' and not r['excluded_reason']]
    calibration = [r for r in included if r['item']['split'] == 'calibration' and r['item']['cohort'] == 'natural']
    if calibration_file:
        frozen = json.loads(Path(calibration_file).read_text(encoding='utf-8'))
        if frozen.get('scoring') != scoring or frozen.get('match_mode') != match_mode:
            raise ValueError('Calibration scoring or match_mode mismatch')
        if frozen.get('threshold_source') != 'calibration_major_f1' or not frozen.get('calibration_keys'):
            raise ValueError('Expected a frozen independent calibration file with partition metadata')
        _check_disjoint(frozen['calibration_keys'], test)
        threshold, mapping = frozen['threshold'], frozen['mapping']
        threshold_source = 'frozen_calibration'
    else:
        mapping = category_mapping(calibration) if scoring == 'categorical' and calibration else None
        threshold_source = 'explicit_diagnostic' if threshold is not None else 'calibration_major_f1'
        if threshold is None:
            if not calibration:
                raise ValueError('Need calibration rows, calibration_file, or explicit diagnostic threshold')
            threshold = best_threshold(calibration, scoring, mapping)
        cal_curve = curve(calibration, scoring, mapping) if calibration else []
        budgets = {}
        for budget in (10, 25, 50, 100):
            affordable = [p for p in cal_curve if p['triggers_per_100'] <= budget]
            if affordable:
                budgets[str(budget)] = max(affordable, key=lambda p: (p['recall'] or 0., -p['triggers_per_100']))['threshold']
        frozen = dict(scoring=scoring, threshold=threshold, mapping=mapping, match_mode=match_mode,
                      threshold_source=threshold_source, budget_thresholds=budgets,
                      calibration_keys=_partition_keys(assigned_cal), calibration_digest=digest(calibration))
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1.000001:
        raise ValueError('threshold must be finite and within [0, 1.000001]')
    if mapping is not None and (set(mapping) != {'low', 'medium', 'high'} or
            any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in mapping.values())):
        raise ValueError('Invalid categorical mapping')
    # Validate scores before doing any expensive reporting or writing output.
    from eval.candidate_error_core import score
    for r in included:
        for span in r['predicted']:
            value = score(span, scoring, mapping)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid normalized prediction confidence')
    reports = {}
    for split in ('calibration', 'test'):
        for cohort in ('natural', 'contrast'):
            selected = [r for r in included if r['item']['split'] == split and r['item']['cohort'] == cohort]
            if selected:
                report = report_rows(selected, scoring, threshold, mapping, n_bootstrap)
                report['calibrated_budget_points'] = {b: dict(threshold=t, **aggregate(selected, t, scoring, mapping))
                    for b, t in frozen.get('budget_thresholds', {}).items()}
                overall = report['overall']
                report['major_penalty_sensitivity'] = {
                    str(weight): overall['fp'] + weight * overall['fn'] for weight in (1, 2, 5)}
                reports[f'{split}.{cohort}'] = report
    for r in rows:
        reason = 'not_selected' if r['item']['split'] == 'excluded' else r['excluded_reason']
        r['evaluation_excluded_reason'] = reason
        r['metrics'] = None if reason else count_item(r, threshold, scoring, mapping)
    assigned = [r for r in rows if r['item']['split'] != 'excluded']
    audit = dict(total=len(rows), unassigned=len(rows)-len(assigned), included=len(included),
                 gold_failed=sum(bool(r['excluded_reason']) for r in assigned),
                 prediction_failed=sum(r['prediction_status'] != 'ok' for r in included),
                 matching_failed=sum(r['matching_status'] == 'failed' for r in included), silver=True)
    root = Path(output_dir)
    source = Path(matching_input).resolve()
    if root.resolve() == source or (source.is_file() and root.resolve() == source.parent):
        raise ValueError('Use an output_dir separate from matching artifacts')
    write_json(root/'calibration.json', frozen)
    write_json(root/'rows.json', included)
    write_jsonl(root/'evaluation_rows.jsonl', rows)
    write_json(root/'report.json', dict(scoring=scoring, threshold=threshold, mapping=mapping,
               threshold_source=threshold_source, audit=audit, match_mode=match_mode, reports=reports,
               matching_input=str(source), matching_digest=matching_digest,
               evaluation_rows_digest=digest(rows),
               split_config=dict(data_ids=data_ids, calibration_data_ids=calibration_data_ids,
                                 test_data_ids=test_data_ids,
                                 split_file=str(split_file) if split_file else None)))
    return rows, reports


if __name__ == '__main__':
    import fire
    fire.Fire(evaluate, serialize=lambda result: None)
