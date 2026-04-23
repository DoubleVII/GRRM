from typing import Union
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
    HarmonyError,
)
from inference.run_oss_SQM import init_oss_model, load_encoding
from inference.prompts import get_prep_notes_prompt


def _block_extractor(response:str) -> dict:
    empty_out = {
        "analysis": None,
        "notes": None,
    }

    response = response.strip()
    if not response:
        return empty_out
    if not response.endswith("```"):
        return empty_out
    response = response[:-3]
    block_start = response.rfind("```markdown")
    if block_start == -1:
        return empty_out
    notes = response[block_start+len("```markdown"):].strip()
    analysis = response[:block_start].strip()
    if not notes:
        return empty_out
    return {"analysis": analysis, "notes": notes}

def _difficulty_extractor(response:str) -> int:
    if not response:
        raise ValueError(f"Difficulty score not found in response")
    response = response.strip()
    lines = response.split("\n")
    for line in lines[::-1]:
        if line.strip().startswith("Difficulty score:"):
            score_text = line.split(":")[-1].strip()
            if not score_text.endswith("/10"):
                raise ValueError(f"Difficulty score format error: {score_text}")
            else:
                score = int(score_text.split("/")[0])
                if score < 0 or score > 10:
                    raise ValueError(f"Difficulty score format error: {score_text}")
                return score
    raise ValueError(f"Difficulty score not found in response")


def no_notes_detact(notes: str) -> bool:
    notes = notes.lower()
    if "no special translation notes needed" in notes:
        return True
    return False

def extract_response(response: str):
    response = response.strip()
    extract_out = _block_extractor(response)
    analysis = extract_out["analysis"]
    notes = extract_out["notes"]
    if not analysis:
        return None
    try:
        difficulty = _difficulty_extractor(analysis)
    except ValueError as e:
        print(f"Difficulty score error: {e}")
        return None
    if not notes:
        return None
    notes_valid = not no_notes_detact(notes)
    return {"analysis": analysis, "notes": notes, "notes_valid": notes_valid, "difficulty": difficulty}



def prepare_vllm_inputs(src_list: list[str], src_langs: list[str], trg_langs: list[str], encoding: HarmonyEncoding, reasoning_effort: str = None):
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
    for src_text, src_lang, trg_lang in zip(src_list, src_langs, trg_langs):
        prompt = get_prep_notes_prompt(src_lang, trg_lang, src_text)
        convo = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.USER, prompt),
        ])

        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        inputs.append({"prompt_token_ids": prefill_ids})
    return inputs

def run_generate(llm, inputs: list[int], sampling_params, encoding: HarmonyEncoding) -> list[Union[dict, None]]:
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
        extracted = extract_response(resp_text)
        if extracted is None:
            results.append(None)
            continue
        results.append({
            "analysis": extracted["analysis"],
            "notes": extracted["notes"],
            "thinking": think_text,
            "response": resp_text,
            "notes_valid": extracted["notes_valid"],
            "difficulty": extracted["difficulty"],
        })
    return results

def func_call(
    src_list: list,
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    temperature: float = 0.4,
    top_p: float = 0.7,
    retry: int = 6,
    model = None,
    model_path: str = "gpt-oss-20b",
    reasoning_effort: str = None,
):
    from vllm import SamplingParams

    n = len(src_list)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * n
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * n
    out_data = {}
    out_data["analysis"] = []
    out_data["notes"] = []
    out_data["response"] = []
    out_data["thinking"] = []
    out_data["notes_valid"] = []
    out_data["difficulty"] = []

    if model is None:
        llm = init_oss_model(model_path)
    else:
        llm = model
    encoding = load_encoding()
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    sampling_params = SamplingParams(max_tokens=8192, temperature=temperature, top_p=top_p, stop_token_ids=stop_token_ids)

    vllm_inputs = prepare_vllm_inputs(src_list, src_langs, trg_langs, encoding, reasoning_effort)
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
            res = {"analysis": None, "notes": None, "thinking": None, "response": None}
        out_data["analysis"].append(res["analysis"])
        out_data["notes"].append(res["notes"])
        out_data["response"].append(res["response"])
        out_data["thinking"].append(res["thinking"])
        out_data["notes_valid"].append(res["notes_valid"])
        out_data["difficulty"].append(res["difficulty"])
    return out_data

if __name__ == "__main__":

    import pandas as pd
    # df = pd.read_parquet("/mnt/nfs06/yangs/data/parquet_data/ranking_distill_data/qwen_tower_sampling_zhen.ranking.rl.test.parquet")
    # output_path = "tower_zhen_testset.oss_note.parquet"
    
    df = pd.read_parquet("/mnt/nfs06/yangs/data/parquet_data/ranking_distill_data/qwen_tower_sampling_zhen.ranking.sft_rl.train.parquet")
    output_path = "tower_zhen_train.oss_note.parquet"

    # assert len(df) == 512
    src_list = df["src_text"].tolist()
    src_lang = df["src_lang"].tolist()
    trg_lang = df["trg_lang"].tolist()
    model = "/home/zfs01/yangs/LLM/openai/gpt-oss-120b"

    out_data = func_call(src_list, src_lang, trg_lang, model_path=model)
    analysis = out_data["analysis"]
    notes = out_data["notes"]
    response = out_data["response"]
    thinking = out_data["thinking"]
    notes_valid = out_data["notes_valid"]
    difficulty = out_data["difficulty"]

    df["analysis"] = analysis
    df["notes"] = notes
    df["response"] = response
    df["thinking"] = thinking
    df["notes_valid"] = notes_valid
    df["difficulty"] = difficulty
    df.to_parquet(output_path)
