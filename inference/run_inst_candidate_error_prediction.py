"""Instruct candidate-error prediction over in-memory candidate dictionaries."""
from inference.candidate_error_prediction import predict_candidates
from utils.hard_protocol import normalize_bool


def predict(items: list[dict], model_path: str, *,
            scoring: str = 'probability', prompt_variant: str = 'auto',
            temperature: float = 0.6, top_p: float = 1.0, max_tokens: int = 8192,
            retry: int = 2, enable_thinking: bool | str = False,
            model=None, vllm_kwargs: dict | None = None,
            base_url: str | list[str] | None = None, workers: int = 8,
            format_feedback: bool = False) -> list[dict]:
    """Predict with the Instruct backend, returning one record per input in order.

    Args:
        items: Candidate dictionaries containing identity, source, translation and languages.
        model_path: Local weights path, or served model name when base_url is supplied.
        scoring: probability (historical normalized confidence), score10, or probability01;
            legacy additionally supports severity and categorical. Validity depends on the variant.
        prompt_variant: Default: auto resolves to compact_analysis.
        temperature: First-attempt temperature; retries increase it by 0.1, capped at 1.
        top_p: Nucleus sampling probability.
        max_tokens: Maximum generated tokens per request.
        retry: Additional attempts for invalid responses.
        enable_thinking: Enable thinking in the Instruct chat template.
        model: Existing local backend model instance; None loads model_path on demand.
        vllm_kwargs: Initialization arguments for the built-in local model loader;
            unused when model supplies the instance.
        base_url: OpenAI-compatible online endpoint or endpoints; None selects local vLLM.
        workers: Concurrent online requests; local inference submits all requests together.
        format_feedback: Add format guidance on online retries; unused for local generation.

    Parsing and validation are selected internally from prompt_variant and scoring.
    """
    enable_thinking = normalize_bool(enable_thinking)
    return predict_candidates(
        items, model_path, backend='inst', scoring=scoring, prompt_variant=prompt_variant,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, retry=retry,
        enable_thinking=enable_thinking, model=model,
        vllm_kwargs=vllm_kwargs, base_url=base_url, workers=workers,
        format_feedback=format_feedback)
