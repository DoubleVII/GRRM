"""Lazy model adapters, keeping artifact validation usable without GPU packages."""
from utils.difficulty_response import parse_prediction_response


def generate(prompts, validators, *, backend, model_path, model=None,
             temperature=0.9, top_p=1.0, max_tokens=8192, retry=3,
             enable_thinking=False, reasoning_effort="medium", vllm_kwargs=None,
             response_format="json"):
    if retry < 0:
        raise ValueError("retry must be nonnegative")
    if not prompts:
        return []
    if model is None and (not isinstance(model_path, str) or not model_path.strip()):
        raise ValueError('model_path must be explicitly provided when no model instance is supplied')
    def parser(text, index):
        return validators[index](parse_prediction_response(text, response_format))

    if backend == "oss":
        from inference.run_oss_SQM import init_oss_model, load_encoding
        from inference.run_oss_diverse_mt import _generate_with_retries, _prepare_inputs
        model = model if model is not None else init_oss_model(model_path, **(vllm_kwargs or {}))
        encoding = load_encoding()
        histories = [[] for _ in prompts]
        pending = list(range(len(prompts)))
        class RecordingModel:
            def generate(self, inputs, sampling_params):
                from inference.run_oss_diverse_mt import _response_text
                print(f"Generation (oss): pending={len(inputs)}/{len(prompts)}; "
                      f"temperature={sampling_params.temperature}", flush=True)
                outputs = model.generate(inputs, sampling_params=sampling_params)
                for index, output in zip(pending, outputs):
                    generated = output.outputs[0]
                    thinking, response = _response_text(generated.token_ids, encoding)
                    histories[index].append(dict(response=response, thinking=thinking,
                        raw_output=getattr(generated, "text", None), valid=False,
                        temperature=sampling_params.temperature, output_tokens=len(generated.token_ids),
                        finish_reason=getattr(generated, "finish_reason", None)))
                return outputs
        def recording_parser(text, index):
            parsed = parser(text, index)
            histories[index][-1]["valid"] = parsed is not None
            if parsed is not None:
                pending.remove(index)
            return parsed
        results = _generate_with_retries(
            RecordingModel(), _prepare_inputs(prompts, encoding, reasoning_effort), recording_parser, encoding,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            retry=retry, parser_with_index=True)
        for result, attempts in zip(results, histories):
            result["attempts"] = attempts
            result["response_format"] = response_format
        return results
    if backend != "inst":
        raise ValueError("Unknown backend")
    from vllm import SamplingParams
    from inference.run_inst_mt import init_inst_model, _render_prompt
    from utils.helpers import split_thinking
    engine = model if model is not None else init_inst_model(model_path, **(vllm_kwargs or {}))
    rendered = [_render_prompt(engine, prompt, enable_thinking) for prompt in prompts]
    results = [dict(parsed=None, response=None, thinking=None, raw_output=None) for _ in prompts]
    histories = [[] for _ in prompts]
    for attempt in range(retry + 1):
        pending = [i for i, result in enumerate(results) if result["parsed"] is None]
        if not pending:
            break
        print(f"Generation (inst): attempt={attempt + 1}/{retry + 1}; "
              f"pending={len(pending)}/{len(prompts)}", flush=True)
        outputs = engine.model.generate([rendered[i] for i in pending], SamplingParams(
            temperature=min(1.0, temperature + 0.1 * attempt), top_p=top_p, max_tokens=max_tokens))
        for i, output in zip(pending, outputs):
            raw = output.outputs[0].text
            thinking, answer = split_thinking(raw)
            parsed = parser(answer, i)
            generated = output.outputs[0]
            histories[i].append(dict(response=answer, thinking=thinking, raw_output=raw,
                valid=parsed is not None, temperature=min(1.0, temperature + 0.1 * attempt),
                output_tokens=len(generated.token_ids)
                    if getattr(generated, "token_ids", None) is not None else None,
                finish_reason=getattr(generated, "finish_reason", None)))
            results[i] = dict(parsed=parsed, response=answer,
                              thinking=thinking, raw_output=raw, attempts=histories[i])
    return [dict(result, response_format=response_format) for result in results]
