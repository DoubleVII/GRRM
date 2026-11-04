"""Prepare/run/analyze small non-thinking span evaluators against saved OSS labels.

Subcommands: prepare, run, summarize. Preparation snapshots full explanations and
splits by SHA256(source), keeping all generators/runs together. Only the retained pairwise prompt is used for inference. Outputs are exclusive and checkpointed per batch.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from eval.analyze_span_proxy import DEFAULT_RUNS, compare, within_source, bootstrap_ci, digest
from eval.span_judge import matching_prompt, validate_pair_decision
from eval.span_matching import maximum_matching, metrics
from eval.span_proxy import score_spans
from utils.difficulty_protocol import same_source


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_items(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def prepare(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rows, references, provenance, counts = [], {}, {}, Counter()
    for model, directory in enumerate(args.run_dirs):
        files = sorted((Path(directory)/'evaluation').glob('*.evaluation.json'))
        if not files:
            raise ValueError(f'No evaluation files: {directory}')
        for path in files:
            payload = json.loads(path.read_text())
            did = payload['config']['data_id']
            if did not in references:
                p = Path(args.annotations_dir)/f'{did}.oss.difficulty_annotations.json'
                annotation = json.loads(p.read_text())
                references[did] = {r['row_id']: r for r in annotation['items']}
                provenance[str(p)] = digest(p)
            provenance[str(path)] = digest(path)
            for item in payload['items']:
                counts['total'] += 1
                if item.get('excluded_reason') or item['span_status'] != 'ok':
                    counts['excluded_or_failed'] += 1
                    continue
                gold = references[did][item['row_id']]
                same_source(item, gold)
                if item['reference_score'] != gold['result']['score']:
                    raise ValueError('Annotation reference score changed')
                group = hashlib.sha256(item['src_text'].encode()).hexdigest()
                split = 'dev' if int(group[:8], 16) % 5 == 0 else 'test'
                predicted, reference = item['prediction_result']['spans'], gold['result']['spans']
                pairs = item['result']['pairs']
                if any(not 0 <= p < len(predicted) or not 0 <= g < len(reference) for p, g in pairs):
                    raise ValueError('Annotation reference span count changed')
                row = {k: item[k] for k in ('data_id', 'row_id', 'src_text', 'src_lang', 'trg_lang')}
                row.update(id=f'{model}:{did}:{item["row_id"]}:{item.get("run_id",0)}',
                    model=model, run_id=item.get('run_id', 0), group=group, split=split,
                    prediction=[{k: s[k] for k in ('source_span', 'explanation')} for s in predicted],
                    reference=[{k: s[k] for k in ('source_span', 'explanation')} for s in reference],
                    acceptance=item.get('acceptance'), pairs=pairs, llm=item['metrics']['f1'],
                    proxy=score_spans(item['src_text'], predicted, reference)['f1'])
                rows.append(row)
    selected = {}
    for split, size in (('dev', args.dev_sources), ('test', args.test_sources)):
        by_dataset = defaultdict(set)
        for r in rows:
            if r['split'] == split:
                by_dataset[r['data_id']].add(r['group'])
        # Sampling uses only source identities, not difficulty labels or results.
        selected[split] = {did: sorted(groups, key=lambda g: hashlib.sha256(('judge-v1'+g).encode()).hexdigest())[:size]
                           if size else sorted(groups) for did, groups in by_dataset.items()}
    with (root/'items.jsonl').open('x') as stream:
        for row in rows:
            row['selected'] = row['group'] in selected[row['split']][row['data_id']]
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    save(root/'manifest.json', dict(version=1, counts=dict(counts), run_dirs=args.run_dirs,
         provenance_sha256=provenance, selected_sources=selected,
         split='SHA256(raw src_text) first 8 hex digits modulo 5: 0=dev, otherwise=test',
         dev_sources_per_dataset=args.dev_sources, test_sources_per_dataset=args.test_sources,
         selected_records=dict(Counter(r['split'] for r in rows if r['selected'])),
         all_records=len(rows)))
    print(json.dumps(dict(rows=len(rows), selected=dict(Counter(r['split'] for r in rows if r['selected'])))), flush=True)


def run(args):
    from inference.hard_generation import generate
    from inference.run_inst_mt import init_inst_model, _render_prompt
    from vllm import SamplingParams
    import importlib.metadata
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_items(Path(args.data_dir)/'items.jsonl') if r['split'] == args.split and (r['selected'] or args.all_sources)]
    if args.limit:
        # Identity-only sampling for profiling; avoid taking the first dataset.
        rows = sorted(rows, key=lambda r: hashlib.sha256(('profile-v1'+r['id']).encode()).hexdigest())[:args.limit]
    if not rows:
        raise ValueError('No benchmark rows')
    if (root/f'{args.split}.pairwise.jsonl').exists():
        raise ValueError('Output exists; choose another output directory')
    config = dict(model_path=args.model_path, variant='pairwise', split=args.split,
        enable_thinking=False, temperature=0., top_p=1., retry=args.retry,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size, data_sha256=digest(Path(args.data_dir)/'items.jsonl'),
        gpu_memory_utilization=args.gpu_memory, max_model_len=args.max_model_len,
        versions={k: importlib.metadata.version(k) for k in ('vllm', 'torch', 'transformers')},
        all_sources=args.all_sources, limit=args.limit,
        runner_sha256=digest(Path(__file__)), prompt_module_sha256=digest(Path(__file__).with_name('span_judge.py')))
    save(root/f'{args.split}.config.json', config)
    started = time.perf_counter()
    engine = init_inst_model(args.model_path, gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        enforce_eager=True, seed=20261104, enable_prefix_caching=True)
    startup = time.perf_counter()-started
    # Warmup is excluded from benchmark; actual prompt prefix caching remains enabled.
    engine.model.generate([_render_prompt(engine, 'Return only the word OK.', False)],
                          SamplingParams(temperature=0, max_tokens=8))
    started = time.perf_counter()
    generated_seconds = 0.
    generated_items = output_tokens = prompt_tokens = initial_requests = 0
    stats = Counter()
    with (root/f'{args.split}.pairwise.jsonl').open('x') as stream:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset+args.batch_size]
            eligible = [r for r in batch if r['prediction'] and r['reference']]
            prompts, validators = [], []
            for r in eligible:
                for p in r['prediction']:
                    for g in r['reference']:
                        prompts.append(matching_prompt(r, [p], [g],
                                                       relaxed=r['acceptance'] == 'relaxed'))
                        validators.append(validate_pair_decision)
            for prompt in prompts:
                prompt_tokens += len(engine.tokenizer.encode(_render_prompt(engine, prompt, False)))
            before = time.perf_counter()
            results = generate(prompts, validators, backend='inst', model_path=args.model_path,
                model=engine, temperature=0., top_p=1., retry=args.retry, enable_thinking=False,
                max_tokens=args.max_tokens,
                response_format='json')
            generated_seconds += time.perf_counter()-before
            generated_items += len(eligible)
            initial_requests += len(prompts)
            iterator = iter(results)
            for r in batch:
                n, m = len(r['prediction']), len(r['reference'])
                if n and m:
                    subresults = [next(iterator) for _ in range(n*m)]
                    valid = all(r['parsed'] is not None for r in subresults)
                    pairs = [[i//m, i % m] for i, r in enumerate(subresults) if r['parsed'] and r['parsed']['match']]
                    generation = dict(parsed=dict(analysis='Independent pair decisions.', pairs=pairs) if valid else None,
                        pair_results=subresults, attempts=[a for r in subresults for a in r.get('attempts', [])],
                        thinking=any(bool(r.get('thinking')) for r in subresults),
                        first_attempt_valid=all(r.get('attempts') and r['attempts'][0]['valid'] for r in subresults))
                else:
                    generation = None
                parsed = generation['parsed'] if generation else dict(analysis='Empty side: deterministic.', pairs=[])
                matching = maximum_matching(parsed['pairs'], n) if parsed else None
                value = metrics(len(matching), n-len(matching), m-len(matching), empty_item=n == m == 0) if matching is not None else None
                status = 'ok' if parsed is not None else 'failed'
                stats[status] += 1
                if generation:
                    attempts = generation.get('attempts', [])
                    stats['first_attempt_valid'] += generation.get('first_attempt_valid', bool(attempts and attempts[0]['valid']))
                    stats['attempts'] += len(attempts)
                    stats['has_native_thinking'] += bool(generation.get('thinking'))
                    stats['length_finishes'] += sum(a.get('finish_reason') == 'length' for a in attempts)
                    output_tokens += sum(a.get('output_tokens') or 0 for a in attempts)
                stream.write(json.dumps(dict(id=r['id'], status=status, result=parsed,
                    matching=matching, metrics=value, generation=generation), ensure_ascii=False)+'\n')
            stream.flush()
            print(f'pairwise: {min(offset+len(batch),len(rows))}/{len(rows)}; {dict(stats)}', flush=True)
    elapsed = time.perf_counter()-started
    save(root/f'{args.split}.pairwise.timing.json', dict(startup_seconds=startup,
        end_to_end_seconds=elapsed, generation_seconds=generated_seconds,
        generated_items=generated_items, deterministic_items=len(rows)-generated_items,
        initial_requests=initial_requests,
        generated_items_per_second=generated_items/generated_seconds if generated_seconds else None,
        amortized_ms_per_generated_item=1000*generated_seconds/generated_items if generated_items else None,
        output_tokens=output_tokens, prompt_tokens_first_attempt=prompt_tokens,
        output_tokens_per_generated_item=output_tokens/generated_items if generated_items else None,
        output_tokens_per_second=output_tokens/generated_seconds if generated_seconds else None,
        stats=dict(stats)))


def evaluate_output(rows, path, bootstrap=0):
    output = {r['id']: r for r in read_items(path)}
    rows = [r for r in rows if r['id'] in output]
    if len(rows) != len(output):
        raise ValueError('Output and input identities differ')
    if not rows:
        raise ValueError('No completed output records')
    y = np.array([r['llm'] for r in rows])
    x = np.array([output[r['id']]['metrics']['f1'] if output[r['id']]['metrics'] else 0. for r in rows])
    ok = np.array([output[r['id']]['status'] == 'ok' for r in rows])
    nonempty = np.array([bool(r['prediction']) and bool(r['reference']) for r in rows])
    all_mask = np.ones(len(rows), dtype=bool)
    def grouped(field):
        return {str(k): compare(x, y, ok & np.array([r[field] == k for r in rows])) for k in sorted({r[field] for r in rows})}
    tp = fp = fn = 0
    for r in rows:
        if output[r['id']]['result'] is None:
            continue
        pred = {tuple(p) for p in output[r['id']]['result']['pairs']}
        gold = {tuple(p) for p in r['pairs']}
        tp += len(pred & gold); fp += len(pred-gold); fn += len(gold-pred)
    proxy = np.array([r['proxy'] for r in rows])
    result = dict(n=len(rows), valid=int(ok.sum()), failed=int((~ok).sum()),
        overall=compare(x, y, ok), nonempty=compare(x, y, ok & nonempty),
        failure_as_zero=compare(x, y, all_mask), exact_f1_accuracy=float(np.mean(np.isclose(x[ok], y[ok]))) if ok.any() else None,
        by_dataset=grouped('data_id'), by_generator=grouped('model'), by_language=grouped('src_lang'),
        edge_metrics=metrics(tp, fp, fn), proxy_overall=compare(proxy, y, all_mask),
        proxy_nonempty=compare(proxy, y, nonempty),
        proxy_within_source=within_source(proxy, y, rows, all_mask),
        within_source=within_source(x, y, rows, ok),
        proxy_by_generator_within_source={str(k): within_source(proxy, y, rows, np.array([r['model']==k for r in rows])) for k in sorted({r['model'] for r in rows})},
        by_generator_within_source={str(k): within_source(x, y, rows, ok & np.array([r['model']==k for r in rows])) for k in sorted({r['model'] for r in rows})})
    if bootstrap:
        result['source_bootstrap_spearman_95ci'] = bootstrap_ci(x, y, rows, ok, bootstrap)
    timing_path = path.with_name(path.stem+'.timing.json')
    if timing_path.exists():
        result['timing'] = json.loads(timing_path.read_text())
    return result


def summarize(args):
    rows = read_items(Path(args.data_dir)/'items.jsonl')
    results = {}
    for directory in args.result_dirs:
        root = Path(directory)
        result = {}
        for path in sorted(root.glob('*.jsonl')):
            if not path.with_name(path.stem+'.timing.json').exists():
                continue  # Only completed, closed outputs are eligible for comparison.
            split, variant = path.stem.split('.', 1)
            if split not in ('dev', 'test'):
                continue
            subset = [r for r in rows if r['split'] == split]
            result[path.stem] = evaluate_output(subset, path, args.bootstrap if split == 'test' else 0)
        results[str(root)] = result
    save(args.output_file, results)
    for directory, result in results.items():
        print(directory, flush=True)
        for k, v in result.items():
            if isinstance(v, dict):
                print(k, json.dumps({key: v[key] for key in ('n','failed','overall','nonempty','edge_metrics')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--run-dirs', nargs='+', default=DEFAULT_RUNS)
    p.add_argument('--annotations-dir', default='/home/nfs06/yangs/data/parquet_data/oss_hard_annotations')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--dev-sources', type=int, default=10)
    p.add_argument('--test-sources', type=int, default=80)
    p.set_defaults(function=prepare)
    p = sub.add_parser('run')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--split', choices=('dev','test'), required=True)
    p.add_argument('--all-sources', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--retry', type=int, default=1)
    p.add_argument('--max-tokens', type=int, default=384)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-num-seqs', type=int, default=128)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--gpu-memory', type=float, default=.55)
    p.set_defaults(function=run)
    p = sub.add_parser('summarize')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--result-dirs', nargs='+', required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--bootstrap', type=int, default=200)
    p.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
