from typing import Union
import warnings
from openai_harmony import (
    HarmonyEncoding,
    SystemContent,
    ReasoningEffort,
    HarmonyError,
    Conversation,
    Message,
    Role,
)
from utils.config import LANG_MAP
from inference.run_oss_SQM import init_oss_model, load_encoding
from inference.run_mt import _block_extractor
from inference.prompts import get_oss_group_post_edit_prompt


def extract_response(response: str):
    response = response.strip()
    post_edit_mt_text = _block_extractor(response)
    if post_edit_mt_text is None:
        return None
    post_edit_mt_text = post_edit_mt_text.strip()
    if not post_edit_mt_text:
        return None
    return post_edit_mt_text


def prepare_vllm_inputs(
    src_list: list[str],
    mt_list: list[list[str]],
    notes_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    encoding: HarmonyEncoding,
    reasoning_effort: str = None,
    prompt_builder=get_oss_group_post_edit_prompt,
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
        prompt = prompt_builder(src_lang, trg_lang, src_text, mt_texts, notes)
        convo = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.USER, prompt),
        ])
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        inputs.append({"prompt_token_ids": prefill_ids})
    return inputs


def run_generate(llm, inputs, sampling_params, encoding: HarmonyEncoding) -> list[Union[dict, None]]:
    outs = llm.generate(inputs, sampling_params=sampling_params)
    results = []
    for o in outs:
        gen = o.outputs[0]
        toks = gen.token_ids
        try:
            entries = encoding.parse_messages_from_completion_tokens(toks, Role.ASSISTANT)
        except HarmonyError as e:
            warnings.warn(f"Ignore HarmonyError: {e}")
            results.append(None)
            continue

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
        post_edit_mt_text = extract_response(resp_text)
        if post_edit_mt_text is None:
            results.append(None)
            continue
        results.append({
            "post_edit_mt": post_edit_mt_text,
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
    max_new_tokens: int = 8192,
    prompt_builder=get_oss_group_post_edit_prompt,
):
    from vllm import SamplingParams

    n = len(src_list)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * n
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * n
    out_data = {}
    out_data["response"] = []
    out_data["thinking"] = []
    out_data["post_edit_mt"] = []

    if model is None:
        llm = init_oss_model(model_path)
    else:
        llm = model
    encoding = load_encoding()
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
    )

    vllm_inputs = prepare_vllm_inputs(
        src_list,
        mt_list,
        notes_list,
        src_langs,
        trg_langs,
        encoding,
        reasoning_effort,
        prompt_builder,
    )
    indices = list(range(n))
    eval_out = [None] * n
    batch_results = run_generate(llm, vllm_inputs, sampling_params, encoding)
    for idx, res in zip(indices, batch_results):
        eval_out[idx] = res
    while retry != 0:
        sampling_params.temperature = min(1.0, sampling_params.temperature + 0.2)
        remaining = [i for i in indices if eval_out[i] is None]
        if remaining:
            remaining_inputs = [vllm_inputs[i] for i in remaining]
            retry_results = run_generate(llm, remaining_inputs, sampling_params, encoding)
            for i, res in zip(remaining, retry_results):
                if res is not None:
                    eval_out[i] = res
            remaining = [i for i in remaining if eval_out[i] is None]
        retry -= 1

    for i in range(n):
        res = eval_out[i]
        if res is None:
            warnings.warn(f"Evaluation failed, src_text: {src_list[i]}")
            res = {"post_edit_mt": None, "response": None, "thinking": None}
        out_data["post_edit_mt"].append(res["post_edit_mt"])
        out_data["response"].append(res["response"])
        out_data["thinking"].append(res["thinking"])
    return out_data


if __name__ == "__main__":
    pass
