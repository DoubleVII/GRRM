"""Instruct prediction prompts; shared variants remain identical across models."""
from inference.candidate_error_prompts import build_prediction_prompt


def build_inst_prompt(item, variant='auto', *, scoring='probability'):
    return build_prediction_prompt(item, 'compact_analysis' if variant == 'auto' else variant,
                                   scoring=scoring)
