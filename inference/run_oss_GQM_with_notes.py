from typing import Union
import warnings
from openai_harmony import (
    HarmonyEncoding,
    SystemContent,
    ReasoningEffort,
    Conversation,
    Message,
    Role,
)
from inference.run_oss_SQM import init_oss_model, load_encoding
from inference.run_oss_GQM import validate_candidate_identifiers, extract_response
from inference.prompts import get_GQM_with_notes_prompt


def prepare_vllm_inputs(
    src_list: list[str],
    mt_list: list[str],
    notes_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    encoding: HarmonyEncoding,
    prompt_format: str = "score",
    add_example: bool = True,
    reasoning_effort: str = None,
):
    inputs = []
    system_content = SystemContent.new()
    if reasoning_effort is not None:
        if reasoning_effort.lower() == "high":
            system_content.reasoning_effort = ReasoningEffort.HIGH
        elif reasoning_effort.lower() == "medium":
            system_content.reasoning_effort = ReasoningEffort.MEDIUM
        elif reasoning_effort.lower() == "low":
            system_content.reasoning_effort = ReasoningEffort.LOW
        else:
            raise ValueError(f"Invalid reasoning_effort: {reasoning_effort}")
    for src_text, mt_texts, notes, src_lang, trg_lang in zip(
        src_list, mt_list, notes_list, src_langs, trg_langs
    ):
        prompt = get_GQM_with_notes_prompt(
            src_lang, trg_lang, src_text, mt_texts,
            prompt_format, add_example, notes=notes,
        )
        convo = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.USER, prompt),
        ])
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        inputs.append({"prompt_token_ids": prefill_ids})
    return inputs


def run_generate(
    llm,
    inputs: list,
    mt_count: list[int],
    sampling_params,
    encoding: HarmonyEncoding,
) -> list[Union[dict, None]]:
    outs = llm.generate(inputs, sampling_params=sampling_params)
    results = []
    for o, mt_cnt in zip(outs, mt_count):
        gen = o.outputs[0]
        toks = gen.token_ids
        entries = encoding.parse_messages_from_completion_tokens(toks, Role.ASSISTANT)
        think_text = None
        resp_text = None
        if len(entries) != 2:
            results.append(None)
            continue
        d0 = entries[0].to_dict()
        c0 = d0.get("content")
        if isinstance(c0, list) and len(c0) > 0 and isinstance(c0[0], dict):
            think_text = c0[0].get("text")
        d1 = entries[1].to_dict()
        c1 = d1.get("content")
        if isinstance(c1, list) and len(c1) > 0 and isinstance(c1[0], dict):
            resp_text = c1[0].get("text")
        if resp_text is None:
            results.append(None)
            continue
        res = extract_response(resp_text, mt_cnt, explicit_analysis=True)
        if res is None:
            results.append(None)
            continue
        results.append({
            "analysis": res["analysis"],
            "scores": res["scores"],
            "thinking": think_text,
            "response": resp_text,
        })
    return results


def func_call(
    src_list: list,
    mt_list: list[list[str]],
    notes_list: list,
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    temperature: float = 0.4,
    top_p: float = 0.7,
    retry: int = 6,
    model=None,
    model_path: str = "gpt-oss-20b",
    reasoning_effort: str = None,
    prompt_format: str = "score",
    add_example: bool = True,
):
    from vllm import SamplingParams

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)
    assert len(src_list) == len(mt_list) == len(src_langs) == len(trg_langs)

    out_data = {}
    out_data["scores"] = []
    out_data["analysis"] = []
    out_data["thinking"] = []

    if model is None:
        llm = init_oss_model(model_path)
    else:
        llm = model
    encoding = load_encoding()
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
    )

    vllm_inputs = prepare_vllm_inputs(
        src_list, mt_list, notes_list, src_langs, trg_langs,
        encoding, prompt_format, add_example, reasoning_effort,
    )
    mt_count = [len(mt_texts) for mt_texts in mt_list]
    n = len(src_list)
    indices = list(range(n))
    ranking_out = [None] * n
    batch_results = run_generate(llm, vllm_inputs, mt_count, sampling_params, encoding)
    for idx, res in zip(indices, batch_results):
        ranking_out[idx] = res
    while retry != 0:
        sampling_params.temperature = min(1.0, sampling_params.temperature + 0.2)
        remaining = [i for i in indices if ranking_out[i] is None]
        if remaining:
            remaining_inputs = [vllm_inputs[i] for i in remaining]
            remaining_mt_count = [mt_count[i] for i in remaining]
            retry_results = run_generate(
                llm, remaining_inputs, remaining_mt_count, sampling_params, encoding,
            )
            for i, res in zip(remaining, retry_results):
                if res is not None:
                    ranking_out[i] = res
            remaining = [i for i in remaining if ranking_out[i] is None]
        retry -= 1

    for i in range(n):
        res = ranking_out[i]
        if res is None:
            warnings.warn(
                f"Evaluation failed, src_text: {src_list[i]}, mt_text: {mt_list[i]}"
            )
            res = {"analysis": None, "scores": None, "thinking": None}
        out_data["scores"].append(res["scores"])
        out_data["analysis"].append(res["analysis"])
        out_data["thinking"].append(res["thinking"])
    return out_data
