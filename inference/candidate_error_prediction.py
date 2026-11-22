"""Shared in-memory prediction and validation for the two model entry points."""
from inference.candidate_error_generation import check_items, run_requests
from inference.candidate_error_prompts import MINOR_VARIANTS, SHORT_VARIANTS
from inference.inst_candidate_error_prompts import build_inst_prompt
from inference.oss_candidate_error_prompts import build_oss_prompt
from utils.candidate_error_protocol import identity
from utils.candidate_error_response import validate_compact


def predict_candidates(items, model_path, *, backend, scoring='probability',
                       prompt_variant='auto', temperature=0.6, top_p=1.0, max_tokens=8192,
                       retry=2, reasoning_effort='medium', enable_thinking=False, think_tag='think',
                       model=None, vllm_kwargs=None, base_url=None,
                       workers=8, format_feedback=False):
    check_items(items)
    if backend not in ('inst', 'oss'):
        raise ValueError('Unknown backend')
    if prompt_variant == 'auto':
        prompt_variant = 'compact_analysis' if backend == 'inst' else 'legacy'
    original = items
    missing = [dict(id=identity(item), item=item, status='failed', result=None,
                    error='Missing candidate translation')
               for item in items if not isinstance(item.get('candidate'), str)]
    items = [item for item in items if isinstance(item.get('candidate'), str)]
    builder = build_inst_prompt if backend == 'inst' else build_oss_prompt
    prompts = [builder(item, prompt_variant, scoring=scoring) for item in items]
    response_format = 'json'
    validator_factory = None
    if prompt_variant != 'legacy':
        response_format = 'candidate_relaxed'
        if prompt_variant != 'legacy_analysis':
            scale = 1 if prompt_variant == 'compact_probability' or scoring == 'probability01' else 10
            allow_minor = prompt_variant in MINOR_VARIANTS + SHORT_VARIANTS
            validator_factory = lambda i: lambda value: validate_compact(
                value, items[i], scale=scale, allow_minor=allow_minor)
    result = run_requests(items, prompts, model_path=model_path, backend=backend,
                          scoring=scoring, temperature=temperature, top_p=top_p,
                          max_tokens=max_tokens, retry=retry, reasoning_effort=reasoning_effort,
                          enable_thinking=enable_thinking, think_tag=think_tag, model=model,
                          vllm_kwargs=vllm_kwargs, base_url=base_url,
                          workers=workers, format_feedback=format_feedback,
                          response_format=response_format, validator_factory=validator_factory)
    by_id = {row['id']: row for row in result + missing}
    return [by_id[identity(item)] for item in original]
