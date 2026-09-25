from typing import Optional

from eval.mt_reranking_eval_core import run_reranking_eval_core
from inference.run_oss_diverse_mt import normalize_bool


def main(
    data_id: tuple[str],
    model_path: str,
    ranking_model_path: str,
    model_name: str,
    data_dir: Optional[str] = None,
    sampling_n: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 8192,
    metrics: list[str] = ["bleurt", "oss"],
    prompt_type: str = "inst-step-by-step",
    ranking_prompt_type: str = "ranking_score",
    ranking_task_type: str = "gqm",
    add_example: bool = False,
    ranking_temperature: Optional[float] = None,
    ranking_top_p: Optional[float] = None,
    ranking_max_new_tokens: Optional[int] = None,
    runs: int = 1,
    save_results: bool = False,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    retry: int = 3,
    enable_thinking: bool = False,
    prompt_version: str = "codeblock",
    **kwargs,
):
    enable_thinking = normalize_bool(enable_thinking, "enable_thinking")
    return run_reranking_eval_core(
        data_id=data_id,
        model_path=model_path,
        ranking_model_path=ranking_model_path,
        model_name=model_name,
        data_dir=data_dir,
        sampling_n=sampling_n,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        metrics=metrics,
        prompt_type=prompt_type,
        ranking_prompt_type=ranking_prompt_type,
        ranking_task_type=ranking_task_type,
        add_example=add_example,
        ranking_temperature=ranking_temperature,
        ranking_top_p=ranking_top_p,
        ranking_max_new_tokens=ranking_max_new_tokens,
        runs=runs,
        save_results=save_results,
        translation_backend="inst",
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        retry=retry,
        enable_thinking=enable_thinking,
        prompt_version=prompt_version,
        **kwargs,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
