"""INST candidate-error prediction over a list of dictionaries."""
from inference.candidate_error_prediction import predict_candidates


def predict(items, model_path, *, scoring='probability', prompt_variant='auto',
            max_tokens=4096, **options):
    return predict_candidates(items, model_path, backend='inst', scoring=scoring,
                              prompt_variant=prompt_variant, max_tokens=max_tokens, **options)
