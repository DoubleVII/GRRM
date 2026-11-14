"""Final-artifact persistence for in-memory candidate-error inference."""
from pathlib import Path

from inference.run_inst_candidate_error_prediction import predict as predict_inst
from inference.run_oss_candidate_error_prediction import predict as predict_oss
from inference.run_oss_candidate_error_annotation import annotate as annotate_items
from utils.candidate_error_protocol import write_json, write_jsonl, read_jsonl


def predict(items, output_path, model_path, *, backend='inst', prompt_variant='auto', **options):
    if backend not in ('inst', 'oss'):
        raise ValueError('Unknown backend')
    rows = (predict_inst if backend == 'inst' else predict_oss)(
        items, model_path, prompt_variant=prompt_variant, **options)
    write_jsonl(output_path, rows)
    for suffix in ('.input_failures.json', '.proposal_failures.json'):
        Path(str(output_path) + suffix).unlink(missing_ok=True)
    return rows


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


def run_requests(items, prompts, output_path, **options):
    """Save completed generic requests used by research drivers."""
    from inference.candidate_error_generation import run_requests as generate
    rows = generate(items, prompts, **options)
    write_jsonl(output_path, rows)
    return rows


def read_annotation_stage(directory, name):
    """Read embedded stage records or the corresponding historical stage artifact."""
    directory = Path(directory)
    final = read_jsonl(directory / 'final.jsonl')
    if not final or any('annotation_stages' in row for row in final):
        return [row['annotation_stages'][name] for row in final
                if name in row.get('annotation_stages', {})]
    return read_jsonl(directory / (name + '.jsonl'))
