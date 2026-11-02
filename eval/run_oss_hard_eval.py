"""Evaluate OSS difficulty predictions against frozen golden annotations.

Uses the same OSS semantic judge and deterministic scoring as run_hard_eval.
"""
from eval.run_hard_eval import main


if __name__ == "__main__":
    import fire
    fire.Fire(main)
