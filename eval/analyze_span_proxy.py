"""Reproduce source-disjoint proxy development and held-out LLM comparison.

Run with .venv/bin/python -m eval.analyze_span_proxy --help. No model is loaded.
Only development rows select parameters; test labels are used for reporting.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

from eval.span_proxy import SpanSource, maximum_weight_matching, pair_score, score_spans
from utils.difficulty_protocol import same_source

DEFAULT_RUNS = [
    'results/hard_eval_gpt-oss-120b.20261104_013927.5qw7zk',
    'results/hard_eval_Qwen3-4B-Instruct-2507.20261103_225955.nekiqV',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(run_dirs, annotations_dir):
    records, provenance, counts = [], {}, Counter()
    annotations = {}
    for model, run_dir in enumerate(run_dirs):
        files = sorted((Path(run_dir) / 'evaluation').glob('*.evaluation.json'))
        if not files:
            raise ValueError(f'No evaluation files: {run_dir}')
        for path in files:
            payload = json.loads(path.read_text())
            did = payload['config']['data_id']
            if did not in annotations:
                annotation_path = Path(annotations_dir) / f'{did}.oss.difficulty_annotations.json'
                data = json.loads(annotation_path.read_text())
                annotations[did] = {x['row_id']: x for x in data['items']}
                provenance[str(annotation_path)] = digest(annotation_path)
            provenance[str(path)] = digest(path)
            for item in payload['items']:
                counts[f'{model}:total'] += 1
                if item.get('excluded_reason') or item['span_status'] != 'ok':
                    counts[f'{model}:excluded_or_failed'] += 1
                    continue
                gold = annotations[did][item['row_id']]
                same_source(item, gold)
                if item['reference_score'] != gold['result']['score']:
                    raise ValueError(f'Reference score drift: {did}, {item["row_id"]}')
                # Recent artifacts save resolved predictions. Older ones are rejected:
                # re-resolving with newer parsing rules could change evaluated spans.
                prediction = item['prediction_result']['spans']
                reference = gold['result']['spans']
                pairs = item['result']['pairs']
                if any(p >= len(prediction) or g >= len(reference) for p, g in pairs):
                    raise ValueError('Reference span count drift')
                # All repeated runs/models and duplicate texts share one split.
                group = hashlib.sha256(item['src_text'].encode()).hexdigest()
                split = 'dev' if int(group[:8], 16) % 5 == 0 else 'test'
                records.append(dict(model=model, data_id=did, row_id=item['row_id'],
                    run_id=item.get('run_id', 0), group=group, split=split,
                    src_text=item['src_text'], src_lang=item['src_lang'],
                    prediction=[s['source_span'] for s in prediction],
                    reference=[s['source_span'] for s in reference],
                    pairs=pairs, llm=item['metrics']['f1'], acceptance=item.get('acceptance')))
    return records, dict(counts), provenance


def configurations():
    configs = {}
    for unit in ('char', 'token'):
        for method in ('exact', 'iou', 'dice'):
            configs[f'{unit}_{method}'] = dict(unit=unit, method=method)
        for gamma in (0.1, 0.25, 0.5, 0.75, 1., 1.5, 2.):
            configs[f'{unit}_containment_{gamma:g}'] = dict(unit=unit, method='containment', gamma=gamma)
        for gamma in (0.5, 0.75, 1.):
            for under_gamma in (0.1, 0.25):
                configs[f'{unit}_asymmetric_{gamma:g}_{under_gamma:g}'] = dict(
                    unit=unit, method='asymmetric', gamma=gamma, under_gamma=under_gamma)
        for power in (0.5, 0.75, 1.5, 2.):
            configs[f'{unit}_containment_0.5_power_{power:g}'] = dict(
                unit=unit, method='containment', gamma=0.5, overlap_power=power)
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            configs[f'{unit}_threshold_{threshold:g}'] = dict(unit=unit, method='threshold', threshold=threshold)
    return configs


def compute(records, configs):
    outputs = {key: [] for key in configs}
    edge_labels, edge_features = [], {key: [] for key in configs}
    grounding = Counter()
    started = time.perf_counter()
    # Source contexts are shared across models and four sampled runs.
    contexts = {}
    for record in records:
        context = contexts.setdefault(record['group'], None)
        if context is None:
            context = contexts[record['group']] = SpanSource(record['src_text'])
        left = [context.ground(q) for q in record['prediction']]
        right = [context.ground(q) for q in record['reference']]
        n, m = len(left), len(right)
        grounding['predicted'] += n
        grounding['ungrounded_predicted'] += sum(not q.occurrences for q in left)
        grounding['reference'] += m
        grounding['ungrounded_reference'] += sum(not q.occurrences for q in right)
        features = {unit: [[context.pair(p, g, unit) for g in right] for p in left]
                    for unit in ('char', 'token')}
        labels = {(p, g) for p, g in record['pairs']}
        if record['split'] == 'test':
            edge_labels.extend((i, j) in labels for i in range(n) for j in range(m))
        for name, config in configs.items():
            options = {k: v for k, v in config.items() if k != 'unit'}
            weights = [[pair_score(*f, **options) for f in row] for row in features[config['unit']]]
            tp = sum(w for _, _, w in maximum_weight_matching(weights))
            outputs[name].append(2*tp/(n+m) if n+m else 1.)
            if record['split'] == 'test':
                edge_features[name].extend(w for row in weights for w in row)
    return ({k: np.asarray(v) for k, v in outputs.items()}, np.array(edge_labels),
            {k: np.asarray(v) for k, v in edge_features.items()}, dict(grounding), time.perf_counter()-started)


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def compare(x, y, mask):
    x, y = x[mask], y[mask]
    return dict(n=len(x), spearman=correlation(x, y),
        pearson=float(pearsonr(x, y).statistic) if len(x) > 2 and np.ptp(x) and np.ptp(y) else None,
        mae=float(np.mean(np.abs(x-y))) if len(x) else None,
        bias=float(np.mean(x-y)) if len(x) else None,
        proxy_mean=float(np.mean(x)) if len(x) else None,
        llm_mean=float(np.mean(y)) if len(x) else None)


def within_source(x, y, records, mask):
    groups = defaultdict(list)
    for i in np.flatnonzero(mask):
        groups[records[i]['group']].append(i)
    concordant = discordant = proxy_tied = judge_tied = 0
    xc, yc = [], []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        a, b = x[indices], y[indices]
        xc.extend(a-a.mean()); yc.extend(b-b.mean())
        for i in range(len(indices)):
            for j in range(i):
                dy, dx = b[i]-b[j], a[i]-a[j]
                if abs(dy) < 1e-10:
                    judge_tied += 1
                elif abs(dx) < 1e-10:
                    proxy_tied += 1
                elif dx*dy > 0:
                    concordant += 1
                else:
                    discordant += 1
    total = concordant + discordant + proxy_tied
    return dict(source_groups=len(groups), llm_unequal_pairs=total,
        concordant=concordant, discordant=discordant, proxy_tied=proxy_tied,
        accuracy_with_half_credit_for_proxy_ties=(concordant+0.5*proxy_tied)/total if total else None,
        centered_pearson=float(pearsonr(xc, yc).statistic) if len(xc) > 1 and np.ptp(xc) and np.ptp(yc) else None,
        judge_tied_pairs=judge_tied)


def bootstrap_ci(x, y, records, mask, repetitions, seed=20261104):
    groups = defaultdict(list)
    for i in np.flatnonzero(mask):
        groups[records[i]['group']].append(i)
    indices = list(groups.values())
    if not indices:
        return None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        selected = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
        value = correlation(x[selected], y[selected])
        if value is not None:
            samples.append(value)
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])] if samples else None


def attack_checks(records, config):
    attacks = defaultdict(list)
    started = time.perf_counter()
    for r in records:
        if r['split'] != 'test' or not r['reference']:
            continue
        source, gold = r['src_text'], r['reference']
        for name, predictions in (('full_source', [source]), ('perfect_reference', gold),
                                  ('duplicate_reference_twice', gold+gold)):
            attacks[name].append(score_spans(source, predictions, gold, **config)['f1'])
    elapsed = time.perf_counter()-started
    return dict(n=len(attacks['full_source']),
        metrics={k: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                         fraction_above_0_8=float(np.mean(np.array(v) > .8))) for k, v in attacks.items()},
        seconds=elapsed)


def benchmark(records, config):
    # Includes grounding, normalization and assignment, with no cross-item cache.
    started = time.perf_counter()
    for r in records:
        score_spans(r['src_text'], r['prediction'], r['reference'], **config)
    elapsed = time.perf_counter()-started
    return dict(n=len(records), seconds=elapsed, microseconds_per_item=elapsed/len(records)*1e6,
                items_per_second=len(records)/elapsed, mode='CPU, serial, source context constructed per item')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', nargs='+', default=DEFAULT_RUNS)
    parser.add_argument('--annotations-dir', default='/home/nfs06/yangs/data/parquet_data/oss_hard_annotations')
    parser.add_argument('--output-dir', default='results/span_proxy_20261104')
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    records, counts, provenance = load_records(args.run_dirs, args.annotations_dir)
    print(f'Loaded {len(records)} valid evaluations', flush=True)
    configs = configurations()
    scores, labels, edges, grounding, elapsed = compute(records, configs)
    print(f'Scored {len(configs)} variants in {elapsed:.2f}s', flush=True)
    y = np.array([r['llm'] for r in records])
    dev = np.array([r['split'] == 'dev' for r in records])
    test = ~dev
    nonempty = np.array([bool(r['prediction']) and bool(r['reference']) for r in records])
    model_masks = {str(m): np.array([r['model'] == m for r in records]) for m in range(len(args.run_dirs))}
    dataset_masks = {did: np.array([r['data_id'] == did for r in records]) for did in sorted({r['data_id'] for r in records})}
    language_masks = {lang: np.array([r['src_lang'] == lang for r in records]) for lang in sorted({r['src_lang'] for r in records})}
    development = {k: compare(v, y, dev) for k, v in scores.items()}
    selected = max(configs, key=lambda k: development[k]['spearman'] or -1)
    # A smooth reward with at least square-root decay is a safer RL candidate
    # than binary thresholds or near-zero length penalties. Select on dev only.
    reward_candidates = [k for k, c in configs.items() if c['method'] in ('dice', 'iou') or
                         c['method'] in ('containment', 'asymmetric') and c['gamma'] >= 0.5]
    reward = max(reward_candidates, key=lambda k: development[k]['spearman'] or -1)
    report = dict(run_dirs=args.run_dirs, counts=counts, grounding=grounding,
        split='SHA256(raw src_text) first 8 hex digits modulo 5: 0=dev, otherwise=test',
        n_dev=int(dev.sum()), n_test=int(test.sum()),
        unique_sources=len({r['group'] for r in records}),
        configurations=configs, development=development, selected=selected, reward=reward,
        selection_criterion='Development Spearman; reward restricts smooth methods to gamma >= 0.5',
        test={k: compare(v, y, test) for k, v in scores.items()},
        test_nonempty={k: compare(v, y, test & nonempty) for k, v in scores.items()},
        by_model={m: {k: compare(v, y, test & mask) for k, v in scores.items()} for m, mask in model_masks.items()},
        by_dataset={did: {k: compare(v, y, test & mask) for k, v in scores.items()} for did, mask in dataset_masks.items()},
        by_language={lang: compare(scores[reward], y, test & mask) for lang, mask in language_masks.items()},
        by_model_within_source={m: within_source(scores[reward], y, records, test & mask) for m, mask in model_masks.items()},
        test_within_source={k: within_source(scores[k], y, records, test) for k in dict.fromkeys(['token_dice', selected, reward])},
        source_bootstrap_95ci={k: bootstrap_ci(scores[k], y, records, test, args.bootstrap) for k in dict.fromkeys(['token_dice', selected, reward])},
        edge_auc={}, transfer={}, leave_dataset_out={}, provenance_sha256=provenance,
        all_variants_compute_seconds=elapsed)
    report['attack_checks'] = attack_checks(records, configs[reward])
    report['benchmark'] = benchmark(records, configs[reward])
    from scipy.stats import rankdata
    pos = int(labels.sum()); neg = len(labels)-pos
    for k, v in edges.items():
        report['edge_auc'][k] = float((rankdata(v)[labels].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
    for m, mask in model_masks.items():
        chosen = max(reward_candidates, key=lambda k: correlation(scores[k][dev & mask], y[dev & mask]) or -1)
        report['transfer'][m] = dict(selected=chosen,
            test_other_model=compare(scores[chosen], y, test & ~mask))
    for did, mask in dataset_masks.items():
        chosen = max(reward_candidates, key=lambda k: correlation(scores[k][dev & ~mask], y[dev & ~mask]) or -1)
        report['leave_dataset_out'][did] = dict(selected=chosen, test=compare(scores[chosen], y, test & mask))
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (out/'scores.jsonl').open('w') as stream:
        for i, r in enumerate(records):
            stream.write(json.dumps(dict(r, proxy={k: float(scores[k][i]) for k in scores}), ensure_ascii=False)+'\n')
    errors = sorted(np.flatnonzero(test & nonempty), key=lambda i: abs(scores[reward][i]-y[i]), reverse=True)[:50]
    (out/'disagreements.json').write_text(json.dumps([dict(records[i], proxy=float(scores[reward][i])) for i in errors], ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(selected=selected, reward=reward, dev=development[reward], test=report['test'][reward],
                         nonempty=report['test_nonempty'][reward], within_source=report['test_within_source'][reward]), indent=2))


if __name__ == '__main__':
    main()
