"""Candidate-error prediction for SFT models over a list of dictionaries."""
from inference.candidate_error_prediction import predict_candidates
from utils.hard_protocol import normalize_bool
from utils.helpers import validate_think_tag


def predict(items, model_path, *, scoring='score10', prompt_variant='critical_micro_fit',
            max_tokens=8192, enable_thinking=False, think_tag='think', **options):
    validate_think_tag(think_tag)
    return predict_candidates(
        items, model_path, backend='inst', scoring=scoring,
        prompt_variant=prompt_variant, max_tokens=max_tokens,
        enable_thinking=normalize_bool(enable_thinking), think_tag=think_tag, **options)
