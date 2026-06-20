from typing import Optional

from eval.mt_gpe_eval_core import run_gpe_eval_core


def main(
    data_id: tuple[str],
    model_path: str,
    gpe_model_path: str,
    model_name: str,
    data_dir: Optional[str] = None,
    sampling_n: int = 4,
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    gpe_temperature: Optional[float] = None,
    gpe_top_p: Optional[float] = None,
    gpe_max_new_tokens: Optional[int] = None,
    metrics: list[str] = ["bleurt", "oss"],
    prompt_type: str = "codeblock-think",
    gqm_prompt_type: str = "ranking_score",
    runs: int = 1,
    save_results: bool = False,
    **kwargs,
):
    return run_gpe_eval_core(
        data_id=data_id,
        model_path=model_path,
        gpe_model_path=gpe_model_path,
        model_name=model_name,
        data_dir=data_dir,
        sampling_n=sampling_n,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        gpe_temperature=gpe_temperature,
        gpe_top_p=gpe_top_p,
        gpe_max_new_tokens=gpe_max_new_tokens,
        metrics=metrics,
        prompt_type=prompt_type,
        gqm_prompt_type=gqm_prompt_type,
        runs=runs,
        save_results=save_results,
        use_notes=False,
        gpe_mode="gqm_gpe",
        **kwargs,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
