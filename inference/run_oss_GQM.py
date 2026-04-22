from typing import Union
import re
import warnings
from tqdm import tqdm
from openai_harmony import (
    HarmonyEncoding,
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    ReasoningEffort,
)
from utils.config import LANG_MAP, candidate_identifiers
from utils.helpers import find_ints_in_string
from inference.run_oss_SQM import init_oss_model, load_encoding


prompt_template = """Given a source text in {} and multiple translation candidates in {}. Perform a step by step analysis and comparison of the translation quality for the candidates. Finally, scoring the candidates with integer scores on a scale from 0 to 10. Output your analysis and comparison first and output the scores in the end line (e.g. `{}`).

Source text:
```
{}
```

{}"""


prompt_template_no_analysis = """Given a source text in {} and multiple translation candidates in {}. Scoring the candidates with integer scores on a scale from 0 to 10 based on the translation quality. Output the scores in the end line (e.g. `{}`).

Source text:
```
{}
```

{}"""

candidate_prompt = """Translation {}:
```
{}
```
"""


def get_prompt(
    source_lang, target_lang, source_text, mt_texts, explicit_analysis: bool = True
):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]
    if len(mt_texts) == 1:
        raise ValueError(f"Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")

    format_example = ",".join(
        [f"{candidate_identifiers[i]}: 0-10" for i in range(len(mt_texts))]
    )
    candidate_prompts = "".join(
        [
            candidate_prompt.format(candidate_identifiers[i], mt_texts[i])
            for i in range(len(mt_texts))
        ]
    )
    if explicit_analysis:
        return prompt_template.format(
            source_lang, target_lang, format_example, source_text, candidate_prompts
        )
    else:
        return prompt_template_no_analysis.format(
            source_lang, target_lang, format_example, source_text, candidate_prompts
        )


def validate_candidate_identifiers(s: str, expected_score_num: int):
    identifier_locations = []
    for identifier in candidate_identifiers[:expected_score_num]:
        identifier_location = s.rfind(identifier)
        if identifier_location == -1:
            return False
        identifier_locations.append(identifier_location)
        if (
            len(identifier_locations) > 1
            and identifier_location <= identifier_locations[-2]
        ):
            return False
    return True


def extract_response(response: str, expected_score_num: int, explicit_analysis: bool = True):
    response = response.strip()
    last_line_index = response.rfind("\n")
    if last_line_index == -1:
        if explicit_analysis:
            print("only score line in response")
            return None
        else:
            last_line_index = 0
            score_line = response
    else:
        score_line = response[last_line_index + 1 :].strip()
    if not validate_candidate_identifiers(score_line, expected_score_num):
        print(f"invalid score line: {score_line}")
        return None
    scores = find_ints_in_string(score_line, expected_score_num)
    if scores is None:
        return None
    return {"analysis": response[:last_line_index].strip(), "scores": scores}


def prepare_vllm_inputs(
    src_list: list[str],
    mt_list: list[str],
    src_langs: list[str],
    trg_langs: list[str],
    encoding: HarmonyEncoding,
    reasoning_effort: str = None,
    explicit_analysis: bool = True,
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
    for src_text, mt_texts, src_lang, trg_lang in zip(
        src_list, mt_list, src_langs, trg_langs
    ):
        prompt = get_prompt(
            src_lang, trg_lang, src_text, mt_texts, explicit_analysis=explicit_analysis
        )
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, system_content),
                Message.from_role_and_content(Role.USER, prompt),
            ]
        )
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        inputs.append({"prompt_token_ids": prefill_ids})

    return inputs


def run_generate(
    llm,
    inputs: list,
    mt_count: list[int],
    sampling_params,
    encoding: HarmonyEncoding,
    explicit_analysis: bool = True,
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
        res = extract_response(resp_text, mt_cnt, explicit_analysis)
        if res is None:
            results.append(None)
            continue
        results.append(
            {
                "analysis": res["analysis"],
                "scores": res["scores"],
                "thinking": think_text,
                "response": resp_text,
            }
        )
    return results


def func_call(
    src_list: list,
    mt_list: list[list[str]],
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    temperature: float = 0.4,
    top_p: float = 0.7,
    retry: int = 6,
    model: str = "gpt-oss-20b",
    reasoning_effort: str = None,
    explicit_analysis: bool = True,
):
    from vllm import LLM, SamplingParams

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)
    assert len(src_list) == len(mt_list) == len(src_langs) == len(trg_langs)

    out_data = {}
    out_data["scores"] = []
    out_data["analysis"] = []
    out_data["thinking"] = []
    llm = init_oss_model(model)
    encoding = load_encoding()
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
    )

    vllm_inputs = prepare_vllm_inputs(
        src_list,
        mt_list,
        src_langs,
        trg_langs,
        encoding,
        reasoning_effort,
        explicit_analysis,
    )
    mt_count = [len(mt_texts) for mt_texts in mt_list]
    n = len(src_list)
    indices = list(range(n))
    ranking_out = [None] * n
    batch_results = run_generate(llm, vllm_inputs, mt_count, sampling_params, encoding, explicit_analysis)
    for idx, res in zip(indices, batch_results):
        ranking_out[idx] = res
    while retry != 0:
        sampling_params.temperature = min(1.0, sampling_params.temperature + 0.2)
        remaining = [i for i in indices if ranking_out[i] is None]
        if remaining:
            remaining_inputs = [vllm_inputs[i] for i in remaining]
            remaining_mt_count = [mt_count[i] for i in remaining]
            retry_results = run_generate(
                llm, remaining_inputs, remaining_mt_count, sampling_params, encoding, explicit_analysis
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


if __name__ == "__main__":

    import pandas as pd
    df = pd.read_parquet("/home/nfs06/yangs/data/parquet_data/ranking_distill_data/qwen_tower_sampling_zhen.ranking.rl.test.parquet")
    assert len(df) == 512
    src_list = df["src_text"].tolist()
    mt_list = df["mt_sampling_text"].tolist()
    src_lang = df["src_lang"].tolist()
    trg_lang = df["trg_lang"].tolist()
    model = "/home/zfs01/yangs/LLM/openai/gpt-oss-120b"

    out_data = func_call(src_list, mt_list, src_lang, trg_lang, model=model)
    scores = out_data["scores"]
    df["oss_ranking_score"] = scores
    df.to_parquet("qwen_tower_sampling_zhen.ranking.rl.test.oss_score.parquet")
