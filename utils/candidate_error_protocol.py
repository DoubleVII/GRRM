"""Candidate-error artifacts and exact Unicode span normalization (no fuzzy repair)."""
import hashlib
import json
import math
from pathlib import Path

VERSION = 'candidate-error-v1'
SOURCE_KEYS = ('src_text', 'candidate', 'src_lang', 'trg_lang')


def identity(item):
    return f"{item['data_id']}:{item['row_id']}:{item['candidate_model']}:{item['candidate_index']}"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def read_predictions(path):
    rows = read_jsonl(path)
    for suffix in ('.proposal_failures.json','.input_failures.json'):
        failures = Path(str(path)+suffix)
        if failures.exists():
            rows += json.loads(failures.read_text())
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prediction identities')
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, rows):
    """Replace a complete artifact after generation; never resume or append requests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                 for row in rows), encoding='utf-8')
    temporary.replace(path)


def locate(span, side, text):
    quote = span.get(f'{side}_span')
    if quote is None:
        if span.get(f'{side}_start') is not None or span.get(f'{side}_end') is not None:
            raise ValueError('null quote has non-null offsets')
        return {f'{side}_span': None, f'{side}_start': None, f'{side}_end': None}
    if not isinstance(quote, str) or not quote:
        raise ValueError('quote must be nonempty or null')
    starts = [i for i in range(len(text)) if text.startswith(quote, i)]
    if 'occurrence' in span.get(f'{side}_metadata', {}):
        occurrence = span[f'{side}_metadata']['occurrence']
    else:
        occurrence = span.get(f'{side}_occurrence', 0 if len(starts) == 1 else None)
    start = span.get(f'{side}_start')
    if start is None:
        if type(occurrence) is not int or not 0 <= occurrence < len(starts):
            raise ValueError('missing or ambiguous exact quote')
        start = starts[occurrence]
    if type(start) is not int or start not in starts:
        raise ValueError('wrong quote offset')
    end = start + len(quote)
    if span.get(f'{side}_end', end) != end:
        raise ValueError('wrong end offset')
    return {f'{side}_span': quote, f'{side}_start': start, f'{side}_end': end}


def validate(value, item, *, scoring='severity'):
    if not isinstance(value, dict) or not isinstance(value.get('spans'), list):
        return None
    spans = []
    try:
        for s in value['spans']:
            if not isinstance(s, dict) or s.get('severity') not in ('minor', 'major'):
                return None
            out = locate(s, 'source', item['src_text'])
            if out['source_span'] is None and s.get('error_type') != 'addition':
                return None
            if not isinstance(s.get('explanation'), str) or not s['explanation'].strip():
                return None
            if s.get('external_info_need') not in ('likely', 'unlikely', 'unclear'):
                return None
            if s.get('error_type') not in ('meaning', 'omission', 'addition', 'entity', 'terminology', 'grammar', 'other'):
                return None
            out.update({k: s[k] for k in ('severity', 'explanation', 'error_type', 'external_info_need')})
            if scoring == 'probability':
                p = s.get('confidence')
                if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                    return None
                out['confidence'] = float(p)
            elif scoring == 'categorical':
                if s.get('confidence') not in ('low', 'medium', 'high'):
                    return None
                out['confidence'] = s['confidence']
            elif scoring != 'severity':
                raise ValueError('Unknown scoring')
            spans.append(out)
        return {'spans': spans}
    except (ValueError, TypeError, KeyError):
        return None


def source_cluster(item):
    from utils.config import LANG_MAP
    return digest([LANG_MAP.get(item['src_lang'], item['src_lang']),
                   LANG_MAP.get(item['trg_lang'], item['trg_lang']), item['src_text']])
