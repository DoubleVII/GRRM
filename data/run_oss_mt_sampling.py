"""Sample OSS direct translations into the mt_sampling_text Parquet column."""

from pathlib import Path
from typing import Optional
import warnings

import fire
import pandas as pd

from inference.run_oss_direct_mt import run_direct_stage
from inference.run_oss_SQM import init_oss_model


def main(
    data_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    sampling_n: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    retry: int = 3,
    reasoning_effort: Optional[str] = "medium",
    workers: int = 16,
    timeout: float = 300.0,
    vllm_kwargs: Optional[dict] = None,
    deduplicate: bool = True,
):
    """Use base_url for remote inference; omit it to load an offline vLLM model.

    vllm_kwargs forwards model initialization options to offline vLLM.
    Failed parses are omitted, so exhausted retries or deduplication can leave
    fewer than sampling_n candidates (including an empty list).
    """
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    if sampling_n < 1 or workers < 1:
        raise ValueError("sampling_n and workers must be positive")
    if retry < 0 or timeout <= 0 or max_new_tokens < 1:
        raise ValueError("retry must be nonnegative; timeout and max_new_tokens must be positive")
    df = pd.read_parquet(data_path)
    missing = sorted({src_key, src_lang_key, trg_lang_key} - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model = None
    if not df.empty and not base_url:
        model = init_oss_model(
            model_path or "openai/gpt-oss-120b",
            **(vllm_kwargs or {}),
        )
    samples = []
    failures = 0
    if not df.empty:
        print(f"Sampling {data_path}: {len(df)} rows × {sampling_n} samples "
              f"= {len(df) * sampling_n} tasks", flush=True)
        out = run_direct_stage(
            src_list=[value for value in df[src_key] for _ in range(sampling_n)],
            src_langs=[value for value in df[src_lang_key] for _ in range(sampling_n)],
            trg_langs=[value for value in df[trg_lang_key] for _ in range(sampling_n)],
            model=model, model_path=model_path, base_url=base_url, api_key=api_key,
            reasoning_effort=reasoning_effort, temperature=temperature, top_p=top_p,
            max_tokens=max_new_tokens, retry=retry, workers=workers, timeout=timeout,
        )
        translations = out["translations"]
        expected = len(df) * sampling_n
        if len(translations) != expected:
            raise ValueError(f"Expected {expected} translations, got {len(translations)}")
        for offset in range(0, expected, sampling_n):
            candidates = translations[offset:offset + sampling_n]
            failures += sum(value is None for value in candidates)
            candidates = [value for value in candidates if value is not None]
            samples.append(list(dict.fromkeys(candidates)) if deduplicate else candidates)
        print(f"Sampled {len(df)} rows", flush=True)

    if failures:
        warnings.warn(f"Omitted {failures} failed translation samples after retries")
    df["mt_sampling_text"] = samples
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
