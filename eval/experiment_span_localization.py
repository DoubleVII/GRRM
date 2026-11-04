"""Paired OSS semantic/localization experiment using snapshotted predictions.

Identity-only source sampling; frozen heuristic; interleaved deduplicated prompts;
JSONL checkpoint supports resume without re-running successful requests.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from eval.benchmark_span_judges import read_items, save
from eval.analyze_span_proxy import compare, correlation, digest, within_source
from eval.difficulty_eval_core import matching_prompt
from eval.span_matching import maximum_matching, metrics, validate_pairs
from eval.span_proxy import score_spans


def prompt_for(row, policy):
    prediction = dict(row, result=dict(spans=row['prediction']))
    gold = dict(row, result=dict(spans=row['reference']))
    return matching_prompt(prediction, gold, match_policy=policy)


def prepare(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rows = read_items(Path(args.data_dir)/'items.jsonl')
    groups = defaultdict(set)
    for r in rows:
        if r['split'] == 'test':
            groups[r['data_id']].add(r['group'])
    selected = {did: set(sorted(gs, key=lambda g: hashlib.sha256(('localization-v1'+g).encode()).hexdigest())[:args.sources])
                for did, gs in groups.items()}
    rows = [r for r in rows if r['split'] == 'test' and r['group'] in selected[r['data_id']]]
    with (root/'items.jsonl').open('x') as stream:
        for r in rows:
            r['proxy'] = score_spans(r['src_text'], r['prediction'], r['reference'])['f1']
            stream.write(json.dumps(r, ensure_ascii=False)+'\n')
    requests = {}
    for r in rows:
        if not r['prediction'] or not r['reference']:
            continue
        for policy in ('semantic', 'localization'):
            prompt = prompt_for(r, policy)
            key = hashlib.sha256(prompt.encode()).hexdigest()
            requests[key] = dict(key=key, policy=policy, prompt=prompt,
                                 n=len(r['prediction']), m=len(r['reference']))
    with (root/'requests.jsonl').open('x') as stream:
        # Random hash order mixes dataset, generator, and prompt policy.
        for key in sorted(requests):
            stream.write(json.dumps(requests[key], ensure_ascii=False)+'\n')
    save(root/'manifest.json', dict(data_dir=args.data_dir, sources_per_dataset=args.sources,
        selected_sources={k: sorted(v) for k, v in selected.items()},
        counts=dict(Counter(r['data_id'] for r in rows)), n=len(rows),
        requests=len(requests), nonempty=sum(bool(r['prediction']) and bool(r['reference']) for r in rows),
        input_sha256=digest(Path(args.data_dir)/'items.jsonl'),
        parent_manifest=json.loads((Path(args.data_dir)/'manifest.json').read_text()),
        heuristic='frozen score_spans default: char_asymmetric_0.5_0.25'))
    print(json.dumps(dict(rows=len(rows), unique_requests=len(requests))), flush=True)


def run(args):
    from inference.hard_generation import generate
    from inference.run_oss_SQM import init_oss_model
    import importlib.metadata
    root = Path(args.output_dir)
    output = root/args.generations_file
    done = {r['key'] for r in read_items(output)} if output.exists() else set()
    assigned = read_items(root/'requests.jsonl')[args.skip_first:args.stop_after or None]
    requests = [r for r in assigned if r['key'] not in done]
    config = dict(model_path=args.model_path, temperature=0., top_p=1., retry=2,
        reasoning_effort='medium', max_tokens=2048, seed=20261104,
        batch_size=args.batch_size, gpu_memory_utilization=0.92, max_model_len=8192,
        versions={k: importlib.metadata.version(k) for k in ('vllm','torch','transformers')},
        requests_sha256=digest(root/'requests.jsonl'), prompt_sha256=digest(Path('eval/difficulty_eval_core.py')))
    config_path = root/args.generations_file.replace('.jsonl', '.config.json')
    if not config_path.exists():
        save(config_path, dict(config, skip_first=args.skip_first, stop_after=args.stop_after))
    elif json.loads(config_path.read_text()) != dict(config, skip_first=args.skip_first, stop_after=args.stop_after):
        raise ValueError('Config changed; do not mix resumed generations')
    print(f'Requests pending={len(requests)}, completed={len(done)}', flush=True)
    if not requests:
        return
    started = time.perf_counter()
    model = init_oss_model(args.model_path, gpu_memory_utilization=.92, max_model_len=8192,
        max_num_seqs=96, enforce_eager=True, seed=20261104, enable_prefix_caching=True)
    print(f'Startup seconds={time.perf_counter()-started:.1f}', flush=True)
    with output.open('a', buffering=1) as stream:
        for offset in range(0, len(requests), args.batch_size):
            batch = requests[offset:offset+args.batch_size]
            validators = [lambda v, n=r['n'], m=r['m']: validate_pairs(v, n, m) for r in batch]
            before = time.perf_counter()
            results = generate([r['prompt'] for r in batch], validators, backend='oss',
                model_path=args.model_path, model=model, temperature=0., top_p=1., max_tokens=2048,
                retry=2, reasoning_effort='medium')
            for request, result in zip(batch, results):
                stream.write(json.dumps(dict(key=request['key'], policy=request['policy'], generation=result), ensure_ascii=False)+'\n')
            print(f'CHECKPOINT {offset+len(batch)}/{len(requests)}; seconds={time.perf_counter()-before:.1f}; '
                  f'failed={sum(r["parsed"] is None for r in results)}', flush=True)


def summarize(args):
    root = Path(args.output_dir)
    rows = read_items(root/'items.jsonl')
    generations = {}
    for path in sorted(root.glob('generations*.jsonl')):
        for r in read_items(path):
            if r['key'] in generations:
                raise ValueError('Duplicate generation across workers')
            generations[r['key']] = r['generation']
    expected = {r['key'] for r in read_items(root/'requests.jsonl')}
    if generations.keys() != expected:
        raise ValueError(f'Generation coverage mismatch: missing={len(expected-generations.keys())}, '
                         f'unexpected={len(generations.keys()-expected)}')
    outputs, failures = [], Counter()
    for row in rows:
        record = dict(row)
        n, m = len(row['prediction']), len(row['reference'])
        for policy in ('semantic', 'localization'):
            if n and m:
                key = hashlib.sha256(prompt_for(row, policy).encode()).hexdigest()
                parsed = generations.get(key, {}).get('parsed')
            else:
                parsed = dict(pairs=[], analysis='Empty side deterministic')
            if parsed is None:
                failures[policy] += 1
                record[policy] = None
                record[policy+'_pairs'] = None
                continue
            pairs = parsed['pairs']
            matching = maximum_matching(pairs, n)
            record[policy] = metrics(len(matching), n-len(matching), m-len(matching), empty_item=n == m == 0)['f1']
            record[policy+'_pairs'] = pairs
            record[policy+'_analysis'] = parsed['analysis']
        outputs.append(record)
    with (root/'scores.jsonl').open('w') as stream:
        for row in outputs:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    valid = [r for r in outputs if r['semantic'] is not None and r['localization'] is not None]
    x = np.array([r['proxy'] for r in valid])
    old = np.array([r['llm'] for r in valid])
    control = np.array([r['semantic'] for r in valid])
    new = np.array([r['localization'] for r in valid])
    nonempty = np.array([bool(r['prediction']) and bool(r['reference']) for r in valid])
    ltb = np.array([r['data_id'].startswith('ltb') for r in valid])
    def stats(mask):
        return dict(old=compare(x, old, mask), semantic_control=compare(x, control, mask),
            localization=compare(x, new, mask),
            delta_vs_control=(correlation(x[mask], new[mask]) or 0)-(correlation(x[mask], control[mask]) or 0),
            changed_vs_old=int(np.sum(new[mask] != old[mask])),
            increased_vs_control=int(np.sum(new[mask] > control[mask])),
            decreased_vs_control=int(np.sum(new[mask] < control[mask])))
    all_mask = np.ones(len(valid), dtype=bool)
    report = dict(n_selected=len(rows), n_paired_valid=len(valid), failures=dict(failures),
        generated_unique=len(generations), expected_unique=json.loads((root/'manifest.json').read_text())['requests'],
        overall=stats(all_mask), nonempty=stats(nonempty), ltb=stats(ltb), ltb_nonempty=stats(ltb & nonempty),
        by_dataset={did: stats(np.array([r['data_id']==did for r in valid])) for did in sorted({r['data_id'] for r in valid})},
        by_model={str(m): stats(np.array([r['model']==m for r in valid])) for m in sorted({r['model'] for r in valid})},
        within_source=dict(old=within_source(x, old, valid, all_mask), control=within_source(x, control, valid, all_mask),
                           localization=within_source(x, new, valid, all_mask)))
    # Cluster bootstrap: keep all models/runs from one source in the same resample.
    report['bootstrap_delta_95ci'] = {}
    for name, mask in [('all',all_mask), ('ltb',ltb), ('ltb_nonempty',ltb & nonempty)]:
        groups = defaultdict(list)
        for i in np.flatnonzero(mask):
            groups[valid[i]['group']].append(i)
        groups = list(groups.values())
        rng = np.random.default_rng(20261104)
        deltas, old_deltas = [], []
        for _ in range(1000):
            ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            a,b,c = correlation(x[ix],new[ix]),correlation(x[ix],control[ix]),correlation(x[ix],old[ix])
            if a is not None and b is not None and c is not None:
                deltas.append(a-b); old_deltas.append(a-c)
        report['bootstrap_delta_95ci'][name] = dict(vs_control=np.percentile(deltas,[2.5,97.5]).tolist(),
                                                                   vs_old=np.percentile(old_deltas,[2.5,97.5]).tolist())
    # Edge transitions where quotes are identical: directly probes the proposed confound.
    transitions = Counter()
    changed = []
    for r in valid:
        previous = set(map(tuple, r['semantic_pairs']))
        current = set(map(tuple, r['localization_pairs']))
        historical = set(map(tuple, r['pairs']))
        for i,p in enumerate(r['prediction']):
            for j,g in enumerate(r['reference']):
                if p['source_span'] == g['source_span']:
                    transitions[f'control_{int((i,j) in previous)}_localization_{int((i,j) in current)}'] += 1
                    transitions[f'old_{int((i,j) in historical)}_localization_{int((i,j) in current)}'] += 1
        if previous != current:
            changed.append(r)
    report['identical_quote_edge_transitions'] = dict(transitions)
    (root/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (root/'changed_examples.json').write_text(json.dumps(changed[:150],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('n_selected','n_paired_valid','failures','overall','ltb','ltb_nonempty','bootstrap_delta_95ci','identical_quote_edge_transitions')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','summarize'])
    parser.add_argument('--data-dir', default='results/span_judges_20261104/data')
    parser.add_argument('--output-dir', default='results/span_localization_20261104')
    parser.add_argument('--sources', type=int, default=32)
    parser.add_argument('--model-path', default='/home/zfs01/yangs/LLM/openai/gpt-oss-120b')
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--generations-file', default='generations.jsonl')
    parser.add_argument('--skip-first', type=int, default=0)
    parser.add_argument('--stop-after', type=int, default=0)
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
