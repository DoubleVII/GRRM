"""Explicit post-retry fallback; raw answers remain in generation history."""
import copy
import json

from utils.difficulty_response import parse_prediction_response


def valid_sample_score(value):
    return (isinstance(value, dict) and type(value.get('score')) is int
            and 0 <= value['score'] <= 10)


def judge_spans(value):
    """Keep every supplied span; expose malformed entries as text to the judge."""
    spans = value.get('spans', [])
    if spans is None:
        spans = []
    if not isinstance(spans, list):
        spans = [spans]
    output = []
    for span in spans:
        if isinstance(span, dict):
            quote = span.get('source_span')
            explanation = span.get('explanation', '')
            # Preserve nonstandard span fields as evidence rather than drop them.
            if not isinstance(quote, str):
                quote = json.dumps(span, ensure_ascii=False)
            if not isinstance(explanation, str):
                explanation = json.dumps(explanation, ensure_ascii=False)
        else:
            quote = span if isinstance(span, str) else json.dumps(span, ensure_ascii=False)
            explanation = ''
        output.append(dict(span if isinstance(span, dict) else {}, source_span=quote, explanation=explanation))
    return output


def resolve_prediction(prediction, *, failure_mode='strict', limits=None):
    """Strict first; after exhausted retries use latest score-valid JSON, else zero.

    This function never retries, edits raw answers, or changes a model score.
    Existing fallback provenance survives repeated evaluation.
    """
    from inference.run_oss_hard_annotation import validate_annotation
    if failure_mode not in ('strict', 'fallback'):
        raise ValueError('failure_mode must be strict or fallback')
    previous = prediction.get('acceptance')
    strict = validate_annotation(prediction.get('result'), prediction, **(limits or {}))
    if prediction.get('status') == 'ok' and strict is not None and previous not in ('relaxed', 'empty'):
        return dict(prediction, result=strict, acceptance='strict', strict_valid=True)
    if failure_mode == 'strict':
        return dict(prediction, status='failed', acceptance='failed', strict_valid=False)
    if previous in ('relaxed', 'empty') and valid_sample_score(prediction.get('result')):
        return copy.deepcopy(prediction)
    generation = prediction.get('generation') or {}
    response_format = generation.get('response_format', 'json')
    attempts = generation.get('attempts', [])
    # A currently stored result is the latest answer when available. Otherwise
    # search actual retry responses newest first, never using gold evidence.
    candidates = [(None, prediction.get('result'))]
    candidates += [(a.get('attempt', i), parse_prediction_response(a.get('response'), response_format))
                   for i, a in reversed(list(enumerate(attempts)))]
    candidates.append((None, parse_prediction_response(generation.get('response'), response_format)))
    for attempt, value in candidates:
        if valid_sample_score(value):
            result = copy.deepcopy(value)
            result['spans'] = judge_spans(value)
            return dict(prediction, status='ok', result=result, acceptance='relaxed', strict_valid=False,
                        error=None, fallback=dict(reason='strict_retries_exhausted',
                            selected_attempt=attempt, raw_result=copy.deepcopy(value)))
    return dict(prediction, status='ok',
                result=dict(analysis='No score-valid JSON after retries; explicit empty fallback.', score=0, spans=[]),
                acceptance='empty', strict_valid=False, error=None,
                fallback=dict(reason='no_score_valid_json', selected_attempt=None))
