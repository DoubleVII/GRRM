"""Load, jointly match, and split candidate-error artifacts by dataset."""
from pathlib import Path
import json

from inference.run_oss_candidate_error_matching import match
from utils.candidate_error_protocol import write_json, write_jsonl, digest, source_cluster


def _read_upstream(path, prediction=False):
    # Annotation stage histories can be large; retain only the matching inputs.
    def compact(row):
        return {key: row[key] for key in ('id', 'item', 'status', 'result')}
    with path.open(encoding='utf-8') as stream:
        rows = [compact(json.loads(line)) for line in stream if line.strip()]
    if prediction:
        for suffix in ('.proposal_failures.json', '.input_failures.json'):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                rows.extend(compact(row) for row in json.loads(sidecar.read_text(encoding='utf-8')))
    return rows


def parse_ids(value):
    if value is None:
        return None
    values = value.split(',') if isinstance(value, str) else list(value)
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError('data_ids must contain nonempty dataset names')
    values = [v.strip() for v in values]
    if any(Path(v).name != v or v in ('.', '..') for v in values):
        raise ValueError('data_ids must be plain dataset names')
    return list(dict.fromkeys(values))


def audit_rows(rows):
    return dict(total=len(rows), gold_failed=sum(bool(r.get('excluded_reason')) for r in rows),
                prediction_failed=sum(r['prediction_status'] != 'ok' for r in rows),
                matching_failed=sum(r['matching_status'] == 'failed' for r in rows),
                matching_requests=sum('matching' in r for r in rows), silver=True)


def prepare_rows(predictions, silver, output_dir, *, rescue_max_tokens=None, **options):
    """Legacy research-driver adapter; new consumers should read matching.jsonl."""
    rows = match(predictions, silver, **options)
    primary_failed = sum(r['matching_status'] == 'failed' for r in rows)
    if rescue_max_tokens is not None:
        failed = {r['id'] for r in rows if r['matching_status'] == 'failed'}
        if failed:
            rescued = match([r for r in predictions if r['id'] in failed],
                            [r for r in silver if r['id'] in failed],
                            **dict(options, max_tokens=rescue_max_tokens))
            by_id = {r['id']: r for r in rescued}
            for i, row in enumerate(rows):
                if row['id'] in by_id:
                    rows[i] = dict(by_id[row['id']], primary_matching=row['matching'])
    for row in rows:
        item = row['item']
        item.setdefault('family', item['data_id'])
        item.setdefault('cluster', source_cluster(item))
        item.setdefault('split', 'test')
        item.setdefault('cohort', 'natural')
    write_jsonl(Path(output_dir) / 'matching.jsonl', rows)
    audit = dict(audit_rows(rows), primary_matching_failed=primary_failed,
                 matching_rescued=primary_failed-sum(r['matching_status'] == 'failed' for r in rows))
    return [r for r in rows if not r['excluded_reason']], audit


def main(output_dir, model_path=None, predictions_dir=None, silver_dir=None, data_ids=None,
         predictions_file=None, silver_file=None, match_mode='semantic',
         temperature=0.6, top_p=1., max_tokens=8192, retry=3, reasoning_effort='medium',
         vllm_kwargs=None, dry_run=False):
    """Supply either a pair of files or two directories plus explicit data_ids.

    Each directory dataset contains predictions.jsonl or final.jsonl. All selected
    datasets enter one inference call; output order follows prediction file order.
    No resume or overwrite. dry_run validates inputs without loading a model.
    """
    if isinstance(dry_run, str) and dry_run.lower() in ('true', 'false'):
        dry_run = dry_run.lower() == 'true'
    if not isinstance(dry_run, bool):
        raise ValueError('dry_run must be boolean')
    ids = parse_ids(data_ids)
    if predictions_file is not None or silver_file is not None:
        if not predictions_file or not silver_file or predictions_dir or silver_dir or ids:
            raise ValueError('Supply only predictions_file and silver_file in file mode')
        pairs = [(None, Path(predictions_file), Path(silver_file))]
    else:
        if not predictions_dir or not silver_dir or not ids:
            raise ValueError('Directory mode requires predictions_dir, silver_dir and data_ids')
        pairs = [(name, Path(predictions_dir)/name/'predictions.jsonl',
                  Path(silver_dir)/name/'final.jsonl') for name in ids]
    root = Path(output_dir).resolve()
    predictions, silver, groups, sources = [], [], [], []
    for name, pp, gp in pairs:
        if not pp.is_file() or not gp.is_file():
            raise ValueError(f'Missing prediction or silver file: {pp}, {gp}')
        if root == pp.resolve().parent or root == gp.resolve().parent or \
                root in pp.resolve().parents or root in gp.resolve().parents:
            raise ValueError('output_dir must not contain input artifacts')
        pv, gv = _read_upstream(pp, prediction=True), _read_upstream(gp)
        if not pv or not gv:
            raise ValueError(f'Empty prediction or silver file: {pp}, {gp}')
        if name and any(r['item']['data_id'] != name for r in pv + gv):
            raise ValueError(f'data_id differs from requested dataset: {name}')
        predictions.extend(pv)
        silver.extend(gv)
        groups.append((name, [r['id'] for r in pv]))
        sources.append(dict(data_id=name, predictions_file=str(pp.resolve()), silver_file=str(gp.resolve())))
        print(f'Loaded {name or pp.name}: {len(pv)} candidates', flush=True)
    # Exact mode provides the same input validation without loading any model.
    if dry_run:
        rows = match(predictions, silver, match_mode='exact')
        pending = sum(not r['excluded_reason'] and
                      any(s.get('source_start') is not None for s in r['predicted']) and
                      any(s.get('source_start') is not None for s in r['gold']) for r in rows)
        summary = dict(audit_rows(rows), semantic_requests=pending)
        print(summary, flush=True)
        return summary
    destinations = [root/'manifest.json'] + [root/name/'matching.jsonl' if name else root/'matching.jsonl'
                                            for name, _ in groups]
    if any(p.exists() for p in destinations):
        raise ValueError('Matching output already exists; use a new output_dir')
    options = dict(match_mode=match_mode, temperature=temperature, top_p=top_p,
                   max_tokens=max_tokens, retry=retry, reasoning_effort=reasoning_effort,
                   vllm_kwargs=vllm_kwargs)
    fingerprint = digest([[dict(id=r['id'], item=r['item'], status=r['status'], result=r['result'])
                           for r in values] for values in (predictions, silver)])
    rows = match(predictions, silver, model_path=model_path, **options)
    by_id = {r['id']: r for r in rows}
    files = []
    for name, keys in groups:
        selected = [by_id[key] for key in keys]
        directory = root/name if name else root
        write_jsonl(directory/'matching.jsonl', selected)
        summary = audit_rows(selected)
        write_json(directory/'summary.json', summary)
        files.append(dict(data_id=name, file=str((directory/'matching.jsonl').relative_to(root)), **summary))
    write_json(root/'manifest.json', dict(schema='candidate-error-matching-v1',
               model_path=model_path, options=options, sources=sources, input_digest=fingerprint,
               prompt_policy='candidate-location-v1', files=files, **audit_rows(rows)))
    print(f'Saved {len(rows)} matching records to {root}', flush=True)


if __name__ == '__main__':
    import fire
    fire.Fire(main, serialize=lambda result: None)
