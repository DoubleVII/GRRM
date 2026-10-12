import json
import re
from pathlib import Path
from typing import Optional, Union

from inference.oss_key_points_mt_prompts import build_convergent_prompt
from inference.run_oss_SQM import init_oss_model, load_encoding
from inference.run_oss_diverse_mt import _generate_with_retries, _prepare_inputs, normalize_bool
from inference.run_oss_key_points import run_key_points_stage, validate_key_points_result


def extract_final_translation(response: Optional[str]) -> Optional[str]:
    """Accept exactly one unlabeled fenced translation, with no surrounding prose."""
    if not isinstance(response, str):
        return None
    match = re.fullmatch(r"```[ \t]*\r?\n(.*?)\r?\n```", response.strip(), re.DOTALL)
    if match is None:
        return None
    translation = match.group(1).strip()
    if not translation or re.search(r"(?m)^[ \t]*```", translation):
        return None
    return translation


def _normalize_languages(n, src_langs, trg_langs):
    src_langs = [src_langs] * n if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * n if isinstance(trg_langs, str) else trg_langs
    if not (n == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")
    return src_langs, trg_langs


def run_convergent_stage(
    src_list: list[str],
    key_points_results: list[dict],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    model=None,
    model_path: str = "openai/gpt-oss-120b",
    reasoning_effort: Optional[str] = "medium",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 8192,
    retry: int = 3,
) -> dict:
    src_langs, trg_langs = _normalize_languages(len(src_list), src_langs, trg_langs)
    if len(key_points_results) != len(src_list):
        raise ValueError("All input lists must have the same length")
    if any(validate_key_points_result(value, source) is None
           for value, source in zip(key_points_results, src_list)):
        raise ValueError("Convergent stage requires valid key points for every item")
    prompts = [build_convergent_prompt(sl, tl, source, points)
               for source, points, sl, tl in zip(
                   src_list, key_points_results, src_langs, trg_langs)]
    if not src_list:
        return {"prompts": [], "translations": [], "responses": [], "thinking": []}
    llm = init_oss_model(model_path) if model is None else model
    encoding = load_encoding()
    results = _generate_with_retries(
        llm, _prepare_inputs(prompts, encoding, reasoning_effort),
        extract_final_translation, encoding, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, retry=retry,
    )
    return {
        "prompts": prompts,
        "translations": [result["parsed"] for result in results],
        "responses": [result["response"] for result in results],
        "thinking": [result["thinking"] for result in results],
    }


def run_pipeline(
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    model=None,
    model_path: str = "openai/gpt-oss-120b",
    reasoning_effort: Optional[str] = "medium",
    key_points_temperature: float = 1.0,
    key_points_top_p: float = 1.0,
    stage1_max_tokens: int = 8192,
    final_temperature: float = 1.0,
    final_top_p: float = 1.0,
    final_max_tokens: int = 8192,
    retry: int = 3,
    allow_key_points_failure: bool = False,
) -> dict:
    allow_key_points_failure = normalize_bool(allow_key_points_failure, "allow_key_points_failure")
    src_langs, trg_langs = _normalize_languages(len(src_list), src_langs, trg_langs)
    convergent = {key: [None] * len(src_list)
                  for key in ("prompts", "translations", "responses", "thinking")}
    if not src_list:
        return {"key_points": {key: [] for key in (
            "prompts", "key_points", "responses", "thinking")}, "convergent": convergent}
    llm = init_oss_model(model_path) if model is None else model
    key_points = run_key_points_stage(
        src_list, src_langs, trg_langs, model=llm, model_path=model_path,
        reasoning_effort=reasoning_effort, temperature=key_points_temperature,
        top_p=key_points_top_p, max_tokens=stage1_max_tokens, retry=retry,
    )
    valid_indices = [i for i, value in enumerate(key_points["key_points"])
                     if value is not None or allow_key_points_failure]
    if valid_indices:
        result = run_convergent_stage(
            [src_list[i] for i in valid_indices],
            [key_points["key_points"][i] if key_points["key_points"][i] is not None
             else {"key_points": []} for i in valid_indices],
            [src_langs[i] for i in valid_indices], [trg_langs[i] for i in valid_indices],
            model=llm, model_path=model_path, reasoning_effort=reasoning_effort,
            temperature=final_temperature, top_p=final_top_p,
            max_tokens=final_max_tokens, retry=retry,
        )
        for local_index, original_index in enumerate(valid_indices):
            for key in convergent:
                convergent[key][original_index] = result[key][local_index]
    return {"key_points": key_points, "convergent": convergent}


def pipeline_item(output: dict, index: int) -> dict:
    """Serialize both stages with the same names across inference and evaluation."""
    return {
        "key_points": output["key_points"]["key_points"][index],
        "key_points_prompt": output["key_points"]["prompts"][index],
        "key_points_response": output["key_points"]["responses"][index],
        "key_points_thinking": output["key_points"]["thinking"][index],
        "convergent_prompt": output["convergent"]["prompts"][index],
        "two_stage_translation": output["convergent"]["translations"][index],
        "two_stage_response": output["convergent"]["responses"][index],
        "two_stage_thinking": output["convergent"]["thinking"][index],
    }


def main(
    input_path: str,
    output_path: str,
    model_path: str = "openai/gpt-oss-120b",
    max_samples: int = 0,
    reasoning_effort: str = "medium",
    key_points_temperature: float = 1.0,
    key_points_top_p: float = 1.0,
    stage1_max_tokens: int = 8192,
    final_temperature: float = 1.0,
    final_top_p: float = 1.0,
    final_max_tokens: int = 8192,
    retry: int = 3,
    allow_key_points_failure: bool = False,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Translate a parquet file with key-point analysis followed by convergence."""
    settings = {key: value for key, value in locals().items()
                if key not in {"input_path", "output_path", "model_path"}}
    import pandas as pd

    frame = pd.read_parquet(input_path)
    missing = sorted({"src_text", "src_lang", "trg_lang"} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    model = init_oss_model(model_path, gpu_memory_utilization=gpu_memory_utilization,
                           max_model_len=max_model_len) if len(frame) else None
    output = run_pipeline(
        frame["src_text"].tolist(), frame["src_lang"].tolist(), frame["trg_lang"].tolist(),
        model=model, model_path=model_path, reasoning_effort=reasoning_effort,
        key_points_temperature=key_points_temperature, key_points_top_p=key_points_top_p,
        stage1_max_tokens=stage1_max_tokens, final_temperature=final_temperature,
        final_top_p=final_top_p, final_max_tokens=final_max_tokens, retry=retry,
        allow_key_points_failure=allow_key_points_failure,
    )
    items = [{"index": i, "src_text": row["src_text"], "src_lang": row["src_lang"],
              "trg_lang": row["trg_lang"], "ref_text": row.get("trg_text"),
              **pipeline_item(output, i)} for i, row in frame.iterrows()]
    payload = {"method": "key_points_mt", "model_path": model_path,
               "input_path": input_path, "settings": settings, "items": items}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(items)} items to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
