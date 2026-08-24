"""Deprecated evaluator for legacy JSON/code-fence FFGPE checkpoints."""

from eval import run_mt_ffgpe_eval as _evaluation
from inference.legacy.run_mt_ffgpe import func_call as legacy_func_call


def main(*args, **kwargs):
    original = _evaluation.func_call
    _evaluation.func_call = legacy_func_call
    try:
        return _evaluation.main(*args, **kwargs)
    finally:
        _evaluation.func_call = original


if __name__ == "__main__":
    import fire
    fire.Fire(main)
