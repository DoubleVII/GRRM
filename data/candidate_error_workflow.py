"""Final-artifact persistence for in-memory candidate-error inference."""
from pathlib import Path

from data.candidate_error_input import load_items
from inference.run_oss_candidate_error_annotation import annotate as annotate_items
from utils.candidate_error_protocol import digest, identity, write_json, write_jsonl


def predict_directory(input_dir, output_dir, model_path, *, predictor,
                      candidate_model='provided', prompt_variant, scoring, **options):
    """Predict all recursive Parquet inputs once, then restore file ownership.

    Every input is processed on each run. Validate global identities and metadata
    before inference, and validate result coverage before writing any artifacts.
    """
    source, target = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError(f'Missing input directory: {source}')
    if source == target or source in target.parents:
        raise ValueError('output_dir must be outside input_dir')
    paths = sorted(source.rglob('*.parquet'))
    if not paths:
        raise ValueError(f'No Parquet files in {source}')
    items, groups, owners = [], [], {}
    for path in paths:
        relative = path.relative_to(source)
        values = load_items(path, candidate_model=candidate_model, require_reference=False)
        ids = [identity(item) for item in values]
        for key in ids:
            if key in owners:
                raise ValueError(f'Duplicate candidate identity {key}: {owners[key]} and {relative}. '
                                 'Assign distinct data_id values in the input files.')
            owners[key] = str(relative)
        groups.append((relative, ids))
        items.extend(values)
        print(f'Loaded {relative}: {len(values)} candidates', flush=True)
    input_digest = digest(items)
    digest(dict(prompt_variant=prompt_variant, scoring=scoring, **options))
    print(f'Running prediction: {len(paths)} files, {len(items)} candidates in one workflow', flush=True)
    rows = predictor(items, model_path, prompt_variant=prompt_variant, scoring=scoring, **options)
    by_id = {row['id']: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(owners):
        raise ValueError('Result identities do not exactly cover input candidates; no split outputs written')
    files = []
    for relative, ids in groups:
        selected = [by_id[key] for key in ids]
        directory = target / relative.with_suffix('')
        artifact = directory / 'predictions.jsonl'
        write_jsonl(artifact, selected)
        summary = dict(total=len(selected), failed=sum(r['status'] != 'ok' for r in selected),
                       prompt_variant=prompt_variant)
        write_json(directory / 'summary.json', summary)
        files.append(dict(input_file=str(relative), output_file=str(artifact), **summary))
        print(f'Saved {artifact}: {summary["total"]} candidates, {summary["failed"]} failed', flush=True)
    write_json(target / 'merged_summary.json', dict(
        task='prediction', input_dir=str(source), model_path=model_path,
        candidate_model=candidate_model, prompt_variant=prompt_variant, scoring=scoring,
        options=options, input_digest=input_digest, total=len(items),
        failed=sum(r['status'] != 'ok' for r in rows), files=files))


def annotate(items, output_dir, model_path, **options):
    rows = annotate_items(items, model_path, **options)
    root = Path(output_dir)
    write_jsonl(root / 'final.jsonl', rows)
    write_json(root / 'summary.json', dict(
        total=len(rows), failed=sum(row['status'] != 'ok' for row in rows), silver=True,
        prompt_variant=options.get('prompt_variant', 'legacy'),
        workflow=options.get('workflow', 'review'),
        disputed=sum(row.get('disputed', False) for row in rows)))
    return rows
