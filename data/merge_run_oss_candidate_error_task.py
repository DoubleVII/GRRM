"""Aggregate a directory into one OSS workflow, then split artifacts by input file."""
from pathlib import Path

from data.candidate_error_input import load_items
from inference.run_oss_candidate_error_annotation import annotate
from inference.run_oss_candidate_error_prediction import predict
from utils.candidate_error_protocol import digest, identity, write_json, write_jsonl


def main(input_dir, output_dir, model_path, task, candidate_model='provided',
         prompt_variant=None, scoring='probability', review=True, workflow='review',
         reasoning_effort='medium', temperature=0.6, max_tokens=8192, retry=2,
         vllm_kwargs=None):
    """Read all recursive Parquet inputs before inference; keep existing candidate IDs.

    No resume/skip: every input file is processed. Global duplicate identities are
    rejected before inference rather than silently renumbered or merged.
    """
    if task not in ('annotation', 'prediction'):
        raise ValueError('task must be annotation or prediction')
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
        values = load_items(path, candidate_model=candidate_model,
                            require_reference=task == 'annotation')
        ids = [identity(item) for item in values]
        for key in ids:
            if key in owners:
                raise ValueError(f'Duplicate candidate identity {key}: {owners[key]} and {relative}. '
                                 'Assign distinct data_id values in the input files.')
            owners[key] = str(relative)
        groups.append((relative, ids))
        items.extend(values)
        print(f'Loaded {relative}: {len(values)} candidates', flush=True)
    variant = prompt_variant or ('legacy' if task == 'annotation' else 'critical_combined_minor')
    options = dict(reasoning_effort=reasoning_effort, temperature=temperature,
                   max_tokens=max_tokens, retry=retry, vllm_kwargs=vllm_kwargs)
    # Check ALL metadata before model initialization, including samples with missing candidates.
    input_digest = digest(items)
    digest(options)
    print(f'Running {task}: {len(paths)} files, {len(items)} candidates in one workflow', flush=True)
    if task == 'annotation':
        rows = annotate(items, model_path, prompt_variant=variant,
                        review=review, workflow=workflow, **options)
    else:
        rows = predict(items, model_path, prompt_variant=variant, scoring=scoring, **options)
    by_id = {row['id']: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(owners):
        raise ValueError('Result identities do not exactly cover input candidates; no split outputs written')
    files = []
    for relative, ids in groups:
        selected = [by_id[key] for key in ids]
        directory = target / relative.with_suffix('')
        artifact = directory / ('final.jsonl' if task == 'annotation' else 'predictions.jsonl')
        write_jsonl(artifact, selected)
        summary = dict(total=len(selected), failed=sum(r['status'] != 'ok' for r in selected),
                       prompt_variant=variant)
        if task == 'annotation':
            summary.update(silver=True, workflow=workflow,
                           disputed=sum(r.get('disputed', False) for r in selected))
        write_json(directory / 'summary.json', summary)
        files.append(dict(input_file=str(relative), output_file=str(artifact), **summary))
        print(f'Saved {artifact}: {summary["total"]} candidates, {summary["failed"]} failed', flush=True)
    write_json(target / 'merged_summary.json', dict(task=task, input_dir=str(source),
        model_path=model_path, prompt_variant=variant, scoring=scoring if task == 'prediction' else None,
        review=review if task == 'annotation' else None, workflow=workflow if task == 'annotation' else None,
        options=options, input_digest=input_digest, total=len(items),
        failed=sum(r['status'] != 'ok' for r in rows), files=files))


if __name__ == '__main__':
    import fire
    fire.Fire(main, serialize=lambda result: None)
