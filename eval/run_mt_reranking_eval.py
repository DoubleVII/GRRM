from typing import Optional

from eval.mt_reranking_eval_core import run_reranking_eval_core


def main(
    data_id: tuple[str],
    model_path: str,
    ranking_model_path: str,
    model_name: str,
    data_dir: Optional[str] = None,
    sampling_n: int = 4,
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    metrics: list[str] = ["bleurt", "oss"],
    prompt_type: str = "codeblock-think",
    ranking_prompt_type: str = "ranking_score",
    ranking_task_type: str = "gqm",
    add_example: bool = False,
    ranking_temperature: Optional[float] = None,
    ranking_top_p: Optional[float] = None,
    ranking_max_new_tokens: Optional[int] = None,
    runs: int = 1,
    save_results: bool = False,
    **kwargs,
):
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
        use_notes=False,
        **kwargs,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
