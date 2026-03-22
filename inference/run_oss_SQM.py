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
    HarmonyError,
)
from utils.config import LANG_MAP
from utils.helpers import find_int_in_string, get_auto_tp_size



GEMBA_SQM_template = """Score the following translation from {} to {} on a continuous scale from 0 to 100 that starts on "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".

{} Source:
```
{}
```

{} Translation:
```
{}
```

Oupur your analysis first and score (0-100) in the end line.
"""

GEMBA_SQM_REF_template = """Score the following machine translation from {} to {} with respect to the human reference on a continuous scale from 0 to 100 that starts with "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".

{} Source:
```
{}
```

{} Reference:
```
{}
```

{} Translation:
```
{}
```

Oupur your analysis first and score (0-100) in the end line.
"""

def get_prompt(src_lang: str, trg_lang: str, src_text: str, mt_text: str, ref_text = None):
    if len(src_lang) == 2:
        src_lang = LANG_MAP.get(src_lang, src_lang)
    if len(trg_lang) == 2:
        trg_lang = LANG_MAP.get(trg_lang, trg_lang)
    if ref_text is None:
        return GEMBA_SQM_template.format(src_lang, trg_lang ,src_lang, trg_lang, src_text, mt_text)
    else:
        return GEMBA_SQM_REF_template.format(src_lang, trg_lang, src_lang, src_text, trg_lang, ref_text, trg_lang, mt_text)


def extract_score(response: str):
    score_line = response.strip().split("\n")[-1]
    scores = find_int_in_string(score_line)
    if len(scores) == 0:
        return None
    if len(scores) == 1:
        return scores[0]
    if len(scores) == 2 and scores[1] == 100:
        return scores[0]
    return None

def init_oss_model(model: str):
    from vllm import LLM
    tp_size = get_auto_tp_size()
    llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=tp_size)
    print(f"Detected CUDA devices: {tp_size}; tensor_parallel_size={tp_size}")
    return llm

def load_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding

def prepare_vllm_inputs(src_list: list[str], mt_list: list[str], src_langs: list[str], trg_langs: list[str], encoding: HarmonyEncoding, ref_list: list[str] = None):
    if ref_list is None:
        ref_list = [None] * len(src_list)
    inputs = []
    for src_text, mt_text, src_lang, trg_lang, ref_text in zip(src_list, mt_list, src_langs, trg_langs, ref_list):
        prompt = get_prompt(src_lang, trg_lang, src_text, mt_text, ref_text)
        convo = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
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
        score = extract_score(resp_text)
        if score is None:
            results.append(None)
            continue
        results.append({"score": score, "thinking": think_text, "response": resp_text})
    return results


def func_call(
    src_list: list,
    mt_list: list,
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    ref_list: list[str] = None,
    temperature: float = 0.4,
    top_p: float = 0.7,
    retry: int = 6,
    model = None,
    model_path: str = "gpt-oss-20b",
):
    from vllm import SamplingParams

    assert len(src_list) == len(mt_list)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)
    out_data = {}
    out_data["scores"] = []
    out_data["response"] = []
    out_data["thinking"] = []
    
    if model is None:
        llm = init_oss_model(model_path)
    else:
        llm = model
    encoding = load_encoding()
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    sampling_params = SamplingParams(max_tokens=8192, temperature=temperature, top_p=top_p, stop_token_ids=stop_token_ids)
    
    vllm_inputs = prepare_vllm_inputs(src_list, mt_list, src_langs, trg_langs, encoding, ref_list)
    n = len(src_list)
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
            warnings.warn(f"Evaluation failed, src_text: {src_list[i]}, mt_text: {mt_list[i]}")
            res = {"score": None, "thinking": None, "response": None}
        out_data["scores"].append(res["score"])
        out_data["response"].append(res["response"])
        out_data["thinking"].append(res["thinking"])
    return out_data
