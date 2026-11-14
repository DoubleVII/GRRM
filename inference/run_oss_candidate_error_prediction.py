"""OSS candidate-error prediction over a list of dictionaries."""
from inference.candidate_error_prediction import predict_candidates


def predict(items, model_path, *, scoring='probability', prompt_variant='critical_combined_minor',
            max_tokens=8192, **options):
    return predict_candidates(items, model_path, backend='oss', scoring=scoring,
                              prompt_variant=prompt_variant, max_tokens=max_tokens, **options)
