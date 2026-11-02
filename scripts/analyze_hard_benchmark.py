"""Summarize attempt-level parsing and strict span failures from the live benchmark."""
import json
from collections import Counter
from pathlib import Path
from utils.hard_protocol import canonical_quote, parse_json, validate_spans, valid_candidate
from utils.helpers import extract_translation_response
from eval.hard_eval_core import validate_pairs
from scripts.benchmark_hard_pipeline import read_stage


def reasons(value, item, golden=False):
    if value is None:
        return ['invalid_json']
    errors = []
    if not isinstance(value.get('analysis'), str):
        errors.append('analysis_missing')
    if not isinstance(value.get('spans'), list):
        return errors + ['spans_not_list']
    for span in value['spans']:
        if not isinstance(span, dict):
            errors.append('span_not_dict')
            continue
        text = span.get('source_span')
        if not isinstance(text, str) or not text.strip():
            errors.append('source_span_empty')
        elif canonical_quote(text, item['src_text']) is None:
            errors.append('source_quote_mismatch')
        if not isinstance(span.get('explanation'), str) or not span['explanation'].strip():
            errors.append('explanation_empty')
        if golden:
            target = span.get('translation_span')
            if not (target == '' and span.get('error_type') == 'omission'):
                if canonical_quote(target, item['candidate']) is None:
                    errors.append('translation_quote_mismatch')
            if span.get('error_type') not in {'omission', 'addition', 'other'}:
                errors.append('invalid_error_type')
            if target == '' and span.get('error_type') != 'omission':
                errors.append('empty_non_omission')
    return errors or ['other_validation']


def main(root):
    candidates = read_stage(root, 'candidates')
    eligible = [x for x in candidates if valid_candidate(x)]
    report = {}
    failures = []
    for backend in ['inst', 'oss']:
        path = root / f'{backend}.attempts.json'
        if not path.exists():
            continue
        calls = json.loads(path.read_text())
        for stage in dict.fromkeys(x['stage'] for x in calls):
            batches = [x for x in calls if x['stage'] == stage]
            successful = set()
            counts = []
            errors = Counter()
            total_tokens, truncated = 0, 0
            if 'matching' in stage:
                mode = stage.split('.')[0]
                pred = read_stage(root, 'predictions', mode)
                gold = read_stage(root, 'golden')
                active = [(p, g) for p, g in zip(pred, gold)
                          if p['status'] == g['status'] == 'ok'
                          and p['result']['spans'] and g['result']['spans']]
            for call in batches:
                ok = 0
                for r in call['records']:
                    total_tokens += r['output_tokens']
                    truncated += r['finish_reason'] == 'length'
                    value = parse_json(r['response'])
                    if stage == 'candidates':
                        parsed = extract_translation_response(r['response'])
                        why = ['translation_format']
                    elif 'matching' in stage:
                        p, g = active[r['index']]
                        parsed = validate_pairs(value, len(p['result']['spans']), len(g['result']['spans']))
                        why = ['invalid_json' if value is None else 'invalid_pairs']
                    else:
                        item = eligible[r['index']]
                        parsed = validate_spans(value, item, golden=stage == 'golden')
                        why = reasons(value, item, golden=stage == 'golden')
                    if r['response'] is None:
                        why = ['missing_final_response']
                    if parsed is not None:
                        successful.add(r['index'])
                        ok += 1
                    else:
                        errors.update(why)
                        failures.append(dict(stage=stage, attempt=len(counts)+1,
                                             index=r['index'], reasons=why,
                                             response=r['response']))
                counts.append(dict(requests=call['batch'], success=ok))
            report[stage] = dict(total=batches[0]['batch'], final_success=len(successful),
                                  attempts=counts, errors=dict(errors), output_tokens=total_tokens,
                                  truncated=truncated, seconds=sum(x['seconds'] for x in batches))
    dump = lambda name, value: (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    dump('parsing_summary.json', report)
    dump('parsing_failures.json', failures)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1] if len(sys.argv)>1 else 'results/hard_pipeline_20261102'))
