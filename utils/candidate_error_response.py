"""Relaxed prediction envelopes; grounded spans remain mandatory and failures explicit."""
import json
import re
import unicodedata

from utils.candidate_error_protocol import validate


def parse_response(text):
    """Read the last output fence or a final JSON object after free-form analysis.

    Do not salvage a malformed final fence using an earlier example/analysis object.
    JSON syntax remains JSON: no eval, invented fields, or partial list recovery.
    """
    if not isinstance(text, str):
        return None
    fences = list(re.finditer(r'```[ \t]*(?:json)?[ \t]*\r?\n(.*?)```', text, re.I | re.S))
    if fences:
        if '```' in text[fences[-1].end():]:
            return None
        candidates = [fences[-1][1].strip()]
    else:
        if '```' in text:
            return None
        # Only accept an object ending at the response's end (allow Markdown whitespace).
        candidates = [text[i:].strip() for i, c in enumerate(text) if c == '{']
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and any(key(k) == 'spans' for k in value):
            return value
    return None


def key(value):
    return re.sub(r'[\s_\-]+', '', str(value)).casefold()


def fields(value):
    out = {}
    for k, v in value.items():
        k = key(k)
        if k in out:
            raise ValueError('colliding normalized keys')
        out[k] = v
    return out


def _fold(text):
    """Unicode/case/whitespace normalization with a map to original characters."""
    chars, offsets = [], []
    for i, char in enumerate(text):
        for c in unicodedata.normalize('NFKC', char).casefold():
            if not c.isspace():
                chars.append(c)
                offsets.append(i)
    return ''.join(chars), offsets


def grounded_quote(quote, text):
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError('missing quote')
    exact = [i for i in range(len(text)) if text.startswith(quote, i)]
    if len(exact) == 1:
        return quote, None
    if exact:
        raise ValueError('ambiguous quote; include more context')
    folded, positions = _fold(text)
    needle, _ = _fold(quote)
    if not needle:
        raise ValueError('empty normalized quote')
    starts = [i for i in range(len(folded)) if folded.startswith(needle, i)]
    if len(starts) != 1:
        raise ValueError('missing or ambiguous normalized quote')
    a = starts[0]
    start, end = positions[a], positions[a + len(needle) - 1] + 1
    # Never match half of a casefold expansion (e.g. s inside ß).
    if _fold(text[start:end])[0] != needle:
        raise ValueError('partial normalized character')
    canonical = text[start:end]
    if text.count(canonical) != 1:
        raise ValueError('ambiguous canonical quote')
    return canonical, dict(original_quote=quote, source_start=start, source_end=end,
                           normalization='NFKC+casefold+ignore_whitespace')


def validate_compact(value, item, *, scale=10, allow_minor=False):
    if not isinstance(value, dict):
        return None
    try:
        root = fields(value)
        if not isinstance(root.get('spans'), list):
            return None
        spans, normalizations = [], []
        for index, raw in enumerate(root['spans']):
            if not isinstance(raw, dict):
                return None
            s = fields(raw)
            quote, adjustment = grounded_quote(s.get('sourcespan'), item['src_text'])
            score = s.get('score')
            if isinstance(score, str):
                score = float(score.strip())
            if type(score) not in (int, float) or not 0 <= score <= scale:
                return None
            if scale == 10 and int(score) != score:
                return None
            reason = s.get('reason')
            if not isinstance(reason, str) or not reason.strip():
                return None
            severity = s.get('severity') if allow_minor else 'major'
            if allow_minor:
                if not isinstance(severity, str) or severity.strip().casefold() not in ('major', 'minor'):
                    return None
                severity = severity.strip().casefold()
            spans.append(dict(source_span=quote, severity=severity, confidence=score / scale,
                              explanation=reason, error_type='other', external_info_need='unclear'))
            if adjustment:
                normalizations.append(dict(span_index=index, **adjustment))
        normalized = validate({'spans': spans}, item, scoring='probability')
        if normalized is not None:
            normalized['quote_normalizations'] = normalizations
        return normalized
    except (ValueError, TypeError, KeyError, OverflowError):
        return None

def validate_annotation(value, item):
    if not isinstance(value, dict) or not isinstance(value.get('spans'), list):
        return None
    normalized = []
    for raw in value['spans']:
        if not isinstance(raw, dict):
            return None
        impact = raw.get('impact')
        if type(impact) is not int or impact not in (1, 2, 3, 4):
            return None
        severity = 'major' if impact >= 3 else 'minor'
        if 'severity' in raw and raw['severity'] != severity:
            return None
        normalized.append(dict(raw, severity=severity))
    parsed = validate(dict(value, spans=normalized), item, scoring='probability')
    if parsed is None:
        return None
    for raw, span in zip(value['spans'], parsed['spans']):
        impact = raw.get('impact')
        if type(impact) is not int or impact not in (1, 2, 3, 4):
            return None
        span['impact'] = impact
    return parsed
