from typing import Union
import warnings

from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyError,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
)

from inference.prompts import (
    format_GQM_scores,
    get_oss_GQM_post_edit_completion_prompt,
)
from inference.run_mt import _block_extractor
from inference.run_oss_SQM import init_oss_model, load_encoding


POST_EDIT_ANALYSIS_HEADING = "# Post-edit Analysis"
FINAL_TRANSLATION_HEADING = "# Final post-edited translation"


def extract_response(response: str):
    response = response.strip()
    analysis_start = response.find(POST_EDIT_ANALYSIS_HEADING)
    final_start = response.find(FINAL_TRANSLATION_HEADING)
    if analysis_start == -1 or final_start == -1 or final_start <= analysis_start:
        return None

    analysis = response[
        analysis_start + len(POST_EDIT_ANALYSIS_HEADING):final_start
    ].strip()
    if not analysis:
        return None

    post_edit_mt = _block_extractor(response)
    if post_edit_mt is None or not post_edit_mt.strip():
        return None
    return {
        "post_edit_analysis": analysis,
        "post_edit_mt": post_edit_mt.strip(),
    }


def build_combined_response(
    gqm_analysis: str,
    gqm_scores,
    candidate_count: int,
    post_edit_response: str,
) -> str:
    score_line = format_GQM_scores(gqm_scores, candidate_count)
    return f"{gqm_analysis.strip()}\n\n{score_line}\n\n{post_edit_response.strip()}"


def _system_content(reasoning_effort: str = None) -> SystemContent:
    system_content = SystemContent.new()
    if reasoning_effort is None:
        return system_content

    efforts = {
        "high": ReasoningEffort.HIGH,
        "medium": ReasoningEffort.MEDIUM,
        "low": ReasoningEffort.LOW,
    }
    effort = efforts.get(reasoning_effort.lower())
    if effort is None:
        raise ValueError(f"Invalid reasoning_effort: {reasoning_effort}")
    system_content.reasoning_effort = effort
    return system_content


def prepare_vllm_inputs(
    src_list: list[str],
    mt_list: list[list[str]],
    gqm_analysis_list: list[str],
    gqm_scores_list: list,
    src_langs: list[str],
    trg_langs: list[str],
    encoding: HarmonyEncoding,
    reasoning_effort: str = None,
):
    inputs = []
    system_content = _system_content(reasoning_effort)
    for src_text, mt_texts, analysis, scores, src_lang, trg_lang in zip(
        src_list,
        mt_list,
        gqm_analysis_list,
        gqm_scores_list,
        src_langs,
        trg_langs,
    ):
        prompt = get_oss_GQM_post_edit_completion_prompt(
            src_lang,
            trg_lang,
            src_text,
            mt_texts,
            analysis,
            scores,
        )
        conversation = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.USER, prompt),
        ])
        prompt_token_ids = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        inputs.append({"prompt_token_ids": prompt_token_ids})
    return inputs


def run_generate(
    llm,
    inputs,
    sampling_params,
    encoding: HarmonyEncoding,
) -> list[Union[dict, None]]:
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    results = []
    for output in outputs:
        tokens = output.outputs[0].token_ids
        try:
            entries = encoding.parse_messages_from_completion_tokens(
                tokens, Role.ASSISTANT
            )
        except HarmonyError as error:
            warnings.warn(f"Ignore HarmonyError: {error}")
            results.append(None)
            continue

        if len(entries) != 2:
            results.append(None)
            continue
        thinking_content = entries[0].to_dict().get("content")
        response_content = entries[1].to_dict().get("content")
        thinking = (
            thinking_content[0].get("text")
            if isinstance(thinking_content, list)
            and thinking_content
            and isinstance(thinking_content[0], dict)
            else None
        )
        response = (
            response_content[0].get("text")
            if isinstance(response_content, list)
            and response_content
            and isinstance(response_content[0], dict)
            else None
        )
        if response is None:
            results.append(None)
            continue
        extracted = extract_response(response)
        if extracted is None:
            results.append(None)
            continue
        results.append({
            **extracted,
            "thinking": thinking,
            "response": response,
        })
    return results


def func_call(
    src_list: list,
    mt_list: list[list[str]],
    gqm_analysis_list: list[str],
    gqm_scores_list: list,
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    temperature: float = 0.4,
    top_p: float = 0.7,
    retry: int = 6,
    model=None,
    model_path: str = "gpt-oss-20b",
    reasoning_effort: str = None,
):
    from vllm import SamplingParams

    item_count = len(src_list)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * item_count
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * item_count
    if not (
        item_count
        == len(mt_list)
        == len(gqm_analysis_list)
        == len(gqm_scores_list)
        == len(src_langs)
        == len(trg_langs)
    ):
        raise ValueError("All input lists must have the same length.")

    llm = init_oss_model(model_path) if model is None else model
    encoding = load_encoding()
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=encoding.stop_tokens_for_assistant_actions(),
    )
    inputs = prepare_vllm_inputs(
        src_list,
        mt_list,
        gqm_analysis_list,
        gqm_scores_list,
        src_langs,
        trg_langs,
        encoding,
        reasoning_effort,
    )

    results = run_generate(llm, inputs, sampling_params, encoding)
    for _ in range(retry):
        remaining = [i for i, result in enumerate(results) if result is None]
        if not remaining:
            break
        sampling_params.temperature = min(1.0, sampling_params.temperature + 0.2)
        retry_results = run_generate(
            llm,
            [inputs[i] for i in remaining],
            sampling_params,
            encoding,
        )
        for index, result in zip(remaining, retry_results):
            if result is not None:
                results[index] = result

    out_data = {
        "post_edit_mt": [],
        "post_edit_analysis": [],
        "response": [],
        "thinking": [],
        "combined_response": [],
    }
    for i, result in enumerate(results):
        if result is None:
            warnings.warn(f"Evaluation failed, src_text: {src_list[i]}")
            result = {
                "post_edit_mt": None,
                "post_edit_analysis": None,
                "response": None,
                "thinking": None,
            }
        out_data["post_edit_mt"].append(result["post_edit_mt"])
        out_data["post_edit_analysis"].append(result["post_edit_analysis"])
        out_data["response"].append(result["response"])
        out_data["thinking"].append(result["thinking"])
        combined_response = None
        if result["response"] is not None:
            combined_response = build_combined_response(
                gqm_analysis_list[i],
                gqm_scores_list[i],
                len(mt_list[i]),
                result["response"],
            )
        out_data["combined_response"].append(combined_response)
    return out_data


if __name__ == "__main__":
    pass
