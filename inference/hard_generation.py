"""Lazy model adapters, keeping artifact validation usable without GPU packages."""
from utils.hard_protocol import parse_json


def generate(prompts, validators, *, backend, model_path, model=None,
             temperature=0.0, top_p=1.0, max_tokens=8192, retry=3,
             enable_thinking=False, reasoning_effort="medium", vllm_kwargs=None):
    if retry < 0:
        raise ValueError("retry must be nonnegative")
    if not prompts:
        return []
    def parser(text, index):
        return validators[index](parse_json(text))

    if backend == "oss":
        from inference.run_oss_SQM import init_oss_model, load_encoding
        from inference.run_oss_diverse_mt import _generate_with_retries, _prepare_inputs
        model = model if model is not None else init_oss_model(model_path, **(vllm_kwargs or {}))
        encoding = load_encoding()
        return _generate_with_retries(
            model, _prepare_inputs(prompts, encoding, reasoning_effort), parser, encoding,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            retry=retry, parser_with_index=True)
    if backend != "inst":
        raise ValueError("Unknown backend")
    from vllm import SamplingParams
    from inference.run_inst_mt import init_inst_model, _render_prompt
    from utils.helpers import split_thinking
    engine = model if model is not None else init_inst_model(model_path, **(vllm_kwargs or {}))
    rendered = [_render_prompt(engine, prompt, enable_thinking) for prompt in prompts]
    results = [dict(parsed=None, response=None, thinking=None, raw_output=None) for _ in prompts]
    for attempt in range(retry + 1):
        pending = [i for i, result in enumerate(results) if result["parsed"] is None]
        if not pending:
            break
        outputs = engine.model.generate([rendered[i] for i in pending], SamplingParams(
            temperature=min(1.0, temperature + 0.1 * attempt), top_p=top_p, max_tokens=max_tokens))
        for i, output in zip(pending, outputs):
            raw = output.outputs[0].text
            thinking, answer = split_thinking(raw)
            results[i] = dict(parsed=parser(answer, i), response=answer,
                              thinking=thinking, raw_output=raw)
    return results
