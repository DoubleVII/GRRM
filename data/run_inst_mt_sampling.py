import fire
import pandas as pd

from inference.run_inst_mt import init_inst_model, run_translation_stage


def main(
    data_path: str,
    output_path: str,
    model_path: str,
    src_key: str = "src_text",
    src_lang_key: str = "src_lang",
    trg_lang_key: str = "trg_lang",
    sampling_n: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 8192,
    retry: int = 3,
    enable_thinking: bool = False,
    prompt_version: str = "codeblock",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
    tensor_parallel_size: int = None,
    deduplicate: bool = True,
):
    """Sample translations per input row, optionally removing duplicate candidates."""
    if not output_path.endswith(".parquet"):
        raise ValueError("output_path must end with .parquet")
    if sampling_n < 1:
        raise ValueError("sampling_n must be at least 1")

    df = pd.read_parquet(data_path)
    required = {src_key, src_lang_key, trg_lang_key}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mt_sampling_text = []
    if not df.empty:
        model = init_inst_model(
            model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )
        # Keep each input's samples adjacent so they can be regrouped by slicing.
        out = run_translation_stage(
            src_list=[value for value in df[src_key] for _ in range(sampling_n)],
            src_langs=[value for value in df[src_lang_key] for _ in range(sampling_n)],
            trg_langs=[value for value in df[trg_lang_key] for _ in range(sampling_n)],
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            retry=retry,
            enable_thinking=enable_thinking,
            prompt_version=prompt_version,
        )
        translations = out["translations"]
        expected = len(df) * sampling_n
        if len(translations) != expected:
            raise ValueError(f"Expected {expected} translations, got {len(translations)}")
        for start in range(0, expected, sampling_n):
            candidates = translations[start:start + sampling_n]
            if deduplicate:
                candidates = list(dict.fromkeys(candidates))
            mt_sampling_text.append(candidates)

    df["mt_sampling_text"] = mt_sampling_text
    print(f"Saving to {output_path}")
    df.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(main)
