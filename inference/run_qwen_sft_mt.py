import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from inference.oss_diverse_mt_prompts import (
    build_direct_prompt,
    build_divergent_prompt,
)
from inference.prompts import get_oss_group_post_edit_prompt
from inference.run_mt import _block_extractor
from inference.run_oss_diverse_mt import (
    extract_final_translation,
    extract_json_object,
    validate_divergent_result,
)
from inference.sft_mt_protocol import (
    add_output_instruction,
    build_scd_followup_prompt,
    parse_task_output,
)
from utils.config import LANG_MAP
from utils.helpers import get_auto_tp_size


@dataclass
class SftMtEngine:
    model: object
    tokenizer: object


def init_sft_mt_model(model_path: str, **vllm_kwargs) -> SftMtEngine:
    from vllm import LLM

    tensor_parallel_size = get_auto_tp_size()
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        **vllm_kwargs,
    )
    print(f"Loaded SFT MT model with tensor_parallel_size={tensor_parallel_size}")
    return SftMtEngine(model=model, tokenizer=model.get_tokenizer())


def _normalize_languages(
    size: int,
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
) -> tuple[list[str], list[str]]:
    src_langs = [src_langs] * size if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * size if isinstance(trg_langs, str) else trg_langs
    if not (size == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")
    return src_langs, trg_langs


def _render_messages(engine: SftMtEngine, messages: list[dict]) -> str:
    return engine.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _parse_inner(inner_parser: Callable[[str], object]) -> Callable[[str], object]:
    return lambda text: parse_task_output(text, inner_parser)


def _generate_once(
    engine: SftMtEngine,
    rendered: list[str],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
):
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
    )
    return engine.model.generate(rendered, params)


def generate_validated(
    engine: SftMtEngine,
    messages_list: list[list[dict]],
    inner_parser: Callable[[str], object],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    retry: int,
    n: int = 1,
) -> list[list[dict]]:
    """Generate exactly n parser-valid completions for every conversation."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rendered = [_render_messages(engine, messages) for messages in messages_list]
    parser = _parse_inner(inner_parser)
    valid: list[list[dict]] = [[] for _ in messages_list]

    initial = _generate_once(
        engine,
        rendered,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
    )
    for index, output in enumerate(initial):
        for candidate in output.outputs:
            parsed = parser(candidate.text)
            if parsed is not None:
                valid[index].append({**parsed, "raw_output": candidate.text})

    for attempt in range(1, retry + 1):
        pending = [
            index for index, values in enumerate(valid) for _ in range(n - len(values))
        ]
        if not pending:
            break
        retry_outputs = _generate_once(
            engine,
            [rendered[index] for index in pending],
            temperature=min(1.0, temperature + 0.1 * attempt),
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
        )
        for index, output in zip(pending, retry_outputs):
            candidate = output.outputs[0]
            parsed = parser(candidate.text)
            if parsed is not None and len(valid[index]) < n:
                valid[index].append({**parsed, "raw_output": candidate.text})
    return valid


def run_direct_stage(
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    engine: SftMtEngine,
    sampling_n: int = 1,
    temperature: float = 0.3,
    top_p: float = 0.8,
    max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    src_langs, trg_langs = _normalize_languages(len(src_list), src_langs, trg_langs)
    messages = [[{
        "role": "user",
        "content": add_output_instruction(build_direct_prompt(sl, tl, source)),
    }] for source, sl, tl in zip(src_list, src_langs, trg_langs)]
    outputs = generate_validated(
        engine,
        messages,
        extract_final_translation,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        retry=retry,
        n=sampling_n,
    )
    return {"outputs": outputs, "messages": messages}


def run_scd_pipeline(
    src_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    *,
    engine: SftMtEngine,
    min_candidates: int = 3,
    max_candidates: int = 6,
    stage1_temperature: float = 0.8,
    stage1_top_p: float = 0.95,
    stage1_max_tokens: int = 8192,
    stage2_temperature: float = 0.3,
    stage2_top_p: float = 0.8,
    stage2_max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    stage1_messages = [[{
        "role": "user",
        "content": add_output_instruction(build_divergent_prompt(
            sl, tl, source, min_candidates, max_candidates,
            prompt_type="json", candidate_confidence=False,
        )),
    }] for source, sl, tl in zip(src_list, src_langs, trg_langs)]
    stage1 = generate_validated(
        engine,
        stage1_messages,
        lambda text: validate_divergent_result(extract_json_object(text)),
        temperature=stage1_temperature,
        top_p=stage1_top_p,
        max_tokens=stage1_max_tokens,
        retry=retry,
    )
    valid_indices = [index for index, values in enumerate(stage1) if values]
    stage2_messages = []
    for index in valid_indices:
        source_lang = LANG_MAP.get(src_langs[index], src_langs[index])
        target_lang = LANG_MAP.get(trg_langs[index], trg_langs[index])
        stage2_messages.append([
            *stage1_messages[index],
            {"role": "assistant", "content": stage1[index][0]["raw_output"]},
            {"role": "user", "content": build_scd_followup_prompt(source_lang, target_lang)},
        ])
    stage2_local = generate_validated(
        engine,
        stage2_messages,
        extract_final_translation,
        temperature=stage2_temperature,
        top_p=stage2_top_p,
        max_tokens=stage2_max_tokens,
        retry=retry,
    ) if stage2_messages else []
    stage2 = [[] for _ in src_list]
    messages_by_index = [None for _ in src_list]
    for index, values, messages in zip(valid_indices, stage2_local, stage2_messages):
        stage2[index] = values
        messages_by_index[index] = messages
    return {
        "stage1": stage1,
        "stage1_messages": stage1_messages,
        "stage2": stage2,
        "stage2_messages": messages_by_index,
    }


def run_gpe_pipeline(
    src_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    *,
    engine: SftMtEngine,
    sampling_n: int = 4,
    sampling_temperature: float = 0.8,
    sampling_top_p: float = 0.95,
    sampling_max_tokens: int = 4096,
    post_edit_temperature: float = 0.3,
    post_edit_top_p: float = 0.8,
    post_edit_max_tokens: int = 4096,
    retry: int = 3,
) -> dict:
    sampling = run_direct_stage(
        src_list,
        src_langs,
        trg_langs,
        engine=engine,
        sampling_n=sampling_n,
        temperature=sampling_temperature,
        top_p=sampling_top_p,
        max_tokens=sampling_max_tokens,
        retry=retry,
    )
    valid_indices = [
        index for index, values in enumerate(sampling["outputs"]) if len(values) >= 2
    ]
    post_edit_messages = [[{
        "role": "user",
        "content": add_output_instruction(get_oss_group_post_edit_prompt(
            src_langs[index],
            trg_langs[index],
            src_list[index],
            [value["parsed"] for value in sampling["outputs"][index]],
            None,
        )),
    }] for index in valid_indices]
    post_edit_local = generate_validated(
        engine,
        post_edit_messages,
        _block_extractor,
        temperature=post_edit_temperature,
        top_p=post_edit_top_p,
        max_tokens=post_edit_max_tokens,
        retry=retry,
    ) if post_edit_messages else []
    post_edit = [[] for _ in src_list]
    messages_by_index = [None for _ in src_list]
    for index, values, messages in zip(valid_indices, post_edit_local, post_edit_messages):
        post_edit[index] = values
        messages_by_index[index] = messages
    return {
        "sampling": sampling,
        "post_edit": post_edit,
        "post_edit_messages": messages_by_index,
    }


def first_parsed(outputs: list[list[dict]]) -> list[Optional[str]]:
    return [values[0]["parsed"] if values else None for values in outputs]


def main(
    input_path: str,
    output_path: str,
    method: str,
    model_path: str,
    max_samples: int = 0,
    sampling_n: int = 4,
    retry: int = 3,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
):
    """Run a trained direct, GPE, or SCD model without evaluation."""
    import pandas as pd

    method = method.strip().lower()
    if method not in {"direct", "group_post_edit", "scd"}:
        raise ValueError("method must be one of: direct, group_post_edit, scd")
    frame = pd.read_parquet(input_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0:
        frame = frame.head(max_samples)
    frame = frame.reset_index(drop=True)
    sources = frame["src_text"].tolist()
    src_langs = frame["src_lang"].tolist()
    trg_langs = frame["trg_lang"].tolist()
    engine = init_sft_mt_model(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    if method == "direct":
        pipeline = run_direct_stage(
            sources, src_langs, trg_langs, engine=engine, retry=retry
        )
        final_outputs = pipeline["outputs"]
    elif method == "group_post_edit":
        pipeline = run_gpe_pipeline(
            sources,
            src_langs,
            trg_langs,
            engine=engine,
            sampling_n=sampling_n,
            retry=retry,
        )
        final_outputs = pipeline["post_edit"]
    else:
        pipeline = run_scd_pipeline(
            sources, src_langs, trg_langs, engine=engine, retry=retry
        )
        final_outputs = pipeline["stage2"]

    items = []
    for index, row in frame.iterrows():
        item = {
            "index": index,
            "src_text": row["src_text"],
            "src_lang": row["src_lang"],
            "trg_lang": row["trg_lang"],
            "ref_text": row.get("trg_text"),
            "translation": (
                final_outputs[index][0]["parsed"] if final_outputs[index] else None
            ),
        }
        if method == "direct":
            item["direct"] = final_outputs[index][0] if final_outputs[index] else None
        elif method == "group_post_edit":
            item["direct_candidates"] = pipeline["sampling"]["outputs"][index]
            item["post_edit"] = final_outputs[index][0] if final_outputs[index] else None
        else:
            item["stage1"] = pipeline["stage1"][index][0] if pipeline["stage1"][index] else None
            item["stage2"] = final_outputs[index][0] if final_outputs[index] else None
        items.append(item)
    payload = {
        "method": method,
        "model_path": model_path,
        "input_path": input_path,
        "generation_failures": sum(not values for values in final_outputs),
        "items": items,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items to {destination}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
