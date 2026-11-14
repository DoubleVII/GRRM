"""Detection metrics: one-to-one locations, confidence curves and source-cluster CIs."""
from collections import Counter, defaultdict
import math
import numpy as np
from eval.span_matching import maximum_matching


def eligible(span, major=True):
    return span.get('source_start') is not None and (not major or span['severity'] == 'major')


def score(span, scoring, mapping=None):
    if scoring == 'severity':
        return float(span['severity'] == 'major')
    if scoring == 'categorical':
        return (mapping or {'low': 0., 'medium': .5, 'high': 1.})[span['confidence']]
    return span['confidence']


def iou(a, b):
    if a.get('source_start') is None or b.get('source_start') is None:
        return 0.
    overlap = max(0, min(a['source_end'], b['source_end']) - max(a['source_start'], b['source_start']))
    union = max(a['source_end'], b['source_end']) - min(a['source_start'], b['source_start'])
    return overlap / union if union else 0.


def location_edges(predicted, golden, mode='overlap'):
    edges = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(golden):
            if mode == 'exact':
                match = p.get('source_start') is not None and (p['source_start'], p['source_end']) == (g['source_start'], g['source_end'])
            elif mode == 'overlap':
                # Deterministic diagnostic only; semantic-location graph is primary.
                match = iou(p, g) >= .5
            else:
                raise ValueError(mode)
            if match:
                edges.append([i, j])
    return edges


def prf(tp, fp, fn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def selected(span, scoring, threshold, mapping=None, major=True):
    if not eligible(span, major=False):
        return False
    # Scores estimate actual major error regardless of the separate severity label.
    if scoring == 'severity' and major:
        return span['severity'] == 'major' and score(span, scoring, mapping) >= threshold
    return score(span, scoring, mapping) >= threshold


def count_item(row, threshold, scoring, mapping=None, major=True):
    p, g = row['predicted'], row['gold']
    chosen = [i for i, s in enumerate(p) if selected(s, scoring, threshold, mapping, major)]
    target = [i for i, s in enumerate(g) if eligible(s, major)]
    pi, gi = {old: new for new, old in enumerate(chosen)}, {old: new for new, old in enumerate(target)}
    edges = [[pi[a], gi[b]] for a, b in row['edges'] if a in pi and b in gi]
    tp = len(maximum_matching(edges, len(chosen)))
    chars = set()
    for i in chosen:
        chars.update(range(p[i]['source_start'], p[i]['source_end']))
    return dict(tp=tp, fp=len(chosen)-tp, fn=len(target)-tp, triggers=len(chosen),
                covered_chars=len(chars), source_chars=len(row['item']['src_text']),
                no_error=int(not target), no_error_alarm=int(not target and bool(chosen)),
                valid=int(row.get('prediction_status') == 'ok'))


def aggregate(rows, threshold, scoring, mapping=None, major=True):
    counts = [count_item(r, threshold, scoring, mapping, major) for r in rows]
    c = Counter()
    for value in counts:
        c.update(value)
    return dict(prf(c['tp'], c['fp'], c['fn']), n=len(rows),
        prediction_valid=c['valid'], prediction_failures=len(rows)-c['valid'],
        triggers_per_100=100*c['triggers']/len(rows) if rows else None,
        no_error_n=c['no_error'], no_error_false_alarm_rate=c['no_error_alarm']/c['no_error'] if c['no_error'] else None,
        character_coverage=c['covered_chars']/c['source_chars'] if c['source_chars'] else None)


def curve(rows, scoring, mapping=None):
    values = sorted({score(s, scoring, mapping) for r in rows for s in r['predicted'] if eligible(s, False)}, reverse=True)
    thresholds = [1.000001] + values
    return [dict(threshold=t, **aggregate(rows, t, scoring, mapping)) for t in thresholds]


def best_threshold(rows, scoring, mapping=None):
    points = curve(rows, scoring, mapping)
    return max(points, key=lambda x: (x['f1'] or 0., -x['triggers_per_100'], x['threshold']))['threshold']


def confidence_labels(rows, scoring, mapping=None):
    """High-confidence-first deterministic augmenting matching; TP labels stay fixed."""
    output = []
    for row in rows:
        p, g = row['predicted'], row['gold']
        order = sorted([i for i,s in enumerate(p) if eligible(s, False)],
                       key=lambda i: (-score(p[i], scoring, mapping), i))
        adjacency = defaultdict(list)
        for i, j in row['edges']:
            if eligible(g[j]):
                adjacency[i].append(j)
        owners = {}
        def augment(i, seen):
            for j in sorted(adjacency[i], key=lambda j: (-iou(p[i],g[j]),j)):
                if j in seen:
                    continue
                seen.add(j)
                if j not in owners or augment(owners[j], seen):
                    owners[j] = i
                    return True
            return False
        for i in order:
            success = augment(i, set())
            output.append(dict(id=row['id'], span_index=i, score=score(p[i],scoring,mapping),
                label=int(success), category=p[i].get('confidence'), cluster=row['item']['cluster']))
    return output


def category_mapping(rows):
    labels = confidence_labels(rows, 'categorical')
    # Pool adjacent violators preserves the intended category ordering.
    groups = []
    for c in ('low', 'medium', 'high'):
        ys = [x['label'] for x in labels if x['category'] == c]
        if not ys:
            continue
        groups.append(([c], sum(ys), len(ys)))
        while len(groups)>1 and groups[-2][2] and groups[-1][2] and groups[-2][1]/groups[-2][2] > groups[-1][1]/groups[-1][2]:
            b, a = groups.pop(), groups.pop()
            groups.append((a[0]+b[0], a[1]+b[1], a[2]+b[2]))
    mapping = {}
    for categories, positives, n in groups:
        for c in categories:
            mapping[c] = positives/n
    if not mapping:
        return {'low':0., 'medium':.5, 'high':1.}
    categories = ('low', 'medium', 'high')
    observed = [i for i,c in enumerate(categories) if c in mapping]
    values = [mapping[categories[i]] for i in observed]
    for i,c in enumerate(categories):
        if c not in mapping:
            mapping[c] = float(np.interp(i,observed,values))
    return mapping


def reliability(rows, scoring, mapping=None):
    labels = confidence_labels(rows, scoring, mapping)
    bins = []
    for k in range(10):
        group = [x for x in labels if min(9,int(x['score']*10)) == k]
        if group:
            bins.append(dict(lower=k/10, upper=(k+1)/10, n=len(group),
                mean_score=sum(x['score'] for x in group)/len(group), precision=sum(x['label'] for x in group)/len(group)))
    correlation = None
    if len(bins)>2 and len({b['precision'] for b in bins})>1:
        from scipy.stats import spearmanr
        correlation=float(spearmanr([b['mean_score'] for b in bins],[b['precision'] for b in bins]).statistic)
    span_correlation=None
    reason='constant_score_or_correctness' if labels else 'no_output_spans'
    if len({x['score'] for x in labels})>1 and len({x['label'] for x in labels})>1:
        from scipy.stats import spearmanr
        span_correlation=float(spearmanr([x['score'] for x in labels],[x['label'] for x in labels]).statistic)
        reason=None
    return dict(n=len(labels), brier=sum((x['score']-x['label'])**2 for x in labels)/len(labels) if labels else None,
                ece=sum(b['n']*abs(b['mean_score']-b['precision']) for b in bins)/len(labels) if labels else None,
                bin_precision_spearman=correlation,span_correctness_spearman=span_correlation,
                correlation_reason=reason,unique_scores=len({x['score'] for x in labels}),bins=bins, labels=labels)


def average_precision(points):
    # Interpolated detection AP; all gold, including unproposed errors, remain in recall denominator.
    ordered = sorted([p for p in points if p['recall'] is not None], key=lambda p:p['recall'])
    area, last = 0., 0.
    for i,p in enumerate(ordered):
        area += (p['recall']-last)*max((q['precision'] or 0.) for q in ordered[i:])
        last = p['recall']
    return area if ordered else None


def bootstrap(rows, threshold, scoring, mapping=None, n=2000, seed=20261114):
    if not rows:
        return {}
    clusters = sorted({r['item']['cluster'] for r in rows})
    indices = {k:i for i,k in enumerate(clusters)}
    totals = np.zeros((len(clusters), 4))
    for r in rows:
        c = count_item(r, threshold, scoring, mapping)
        totals[indices[r['item']['cluster']]] += [c['tp'],c['fp'],c['fn'],c['triggers']]
    weights = np.random.default_rng(seed).poisson(1, (n, len(clusters)))
    counts = weights @ totals
    ci = {}
    for key, numerator, denominator in [('precision',counts[:,0],counts[:,0]+counts[:,1]),
        ('recall',counts[:,0],counts[:,0]+counts[:,2]),('f1',2*counts[:,0],2*counts[:,0]+counts[:,1]+counts[:,2])]:
        values = numerator[denominator>0]/denominator[denominator>0]
        ci[key] = np.quantile(values,[.025,.975]).tolist() if len(values) else None
    return ci


def severity_confusion(rows):
    result = Counter()
    for row in rows:
        for a,b in maximum_matching(row['edges'], len(row['predicted'])):
            result[f"{row['gold'][b]['severity']}->{row['predicted'][a]['severity']}"] += 1
    return dict(result)
