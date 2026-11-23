from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from tqdm import tqdm

from inference.inst_mt_prompts import build_translation_prompt
from inference.run_oss_SQM import init_oss_model, load_encoding
from inference.run_oss_diverse_mt import (
    _generate_with_retries,
    _prepare_inputs,
)
from utils.helpers import extract_translation_response


def _run_remote(prompts, *, base_url, model_path, api_key, workers, timeout,
                reasoning_effort, temperature, top_p, max_tokens, retry):
    import httpx

    base_url = base_url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(base_url=base_url + "/", headers=headers, timeout=timeout,
                      trust_env=False) as client:
        if model_path is None:
            response = client.get("models")
            response.raise_for_status()
            models = response.json()["data"]
            if len(models) != 1:
                raise ValueError("Set model_path explicitly when the server does not list exactly one model")
            model_path = models[0]["id"]

        def generate(prompt):
            result = {"parsed": None, "response": None, "thinking": None}
            for attempt in range(retry + 1):
                payload = {
                    "model": model_path,
                    "messages": [{"role": "system", "content": ""},
                                 {"role": "user", "content": prompt}],
                    "temperature": min(1.0, temperature + 0.1 * attempt) if attempt else temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                if reasoning_effort is not None:
                    payload["reasoning_effort"] = reasoning_effort
                try:
                    response = client.post("chat/completions", json=payload)
                    response.raise_for_status()
                except httpx.HTTPError as error:
                    # Invalid parameters/authentication must fail immediately.
                    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500 and error.response.status_code not in (408, 429):
                        raise
                    if attempt == retry:
                        raise
                    continue
                message = response.json()["choices"][0]["message"]
                text = message.get("content")
                result = {
                    "parsed": extract_translation_response(text),
                    "response": text,
                    "thinking": message.get("reasoning") or message.get("reasoning_content"),
                }
                if result["parsed"] is not None:
                    return result
            warnings.warn(f"OSS generation failed after {retry + 1} attempts")
            return result

        results = [None] * len(prompts)
        with tqdm(total=len(prompts), desc="OSS remote sampling", unit="sample") as progress:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(generate, prompt): index
                           for index, prompt in enumerate(prompts)}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    progress.update(1)
        return results


def run_direct_stage(
    src_list: list[str],
    src_langs: Union[str, list[str]],
    trg_langs: Union[str, list[str]],
    *,
    model=None,
    model_path: Optional[str] = None,
    reasoning_effort: Optional[str] = "medium",
    temperature: float = 0.3,
    top_p: float = 0.8,
    max_tokens: int = 4096,
    retry: int = 3,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    workers: int = 16,
    timeout: float = 300.0,
) -> dict:
    """Translate using offline vLLM or an OpenAI-compatible vLLM server.

    With base_url and no model_path, discover the server's sole served model.
    HTTP requests bypass environment proxies for direct access to the server.
    """
    if retry < 0 or workers < 1 or timeout <= 0 or max_tokens < 1:
        raise ValueError("retry must be nonnegative; workers, timeout and max_tokens must be positive")
    if reasoning_effort not in (None, "low", "medium", "high"):
        raise ValueError("reasoning_effort must be one of: low, medium, high, None")
    if base_url and model is not None:
        raise ValueError("model and base_url cannot be used together")
    n = len(src_list)
    src_langs = [src_langs] * n if isinstance(src_langs, str) else src_langs
    trg_langs = [trg_langs] * n if isinstance(trg_langs, str) else trg_langs
    if not (n == len(src_langs) == len(trg_langs)):
        raise ValueError("All input lists must have the same length")

    prompts = [
        build_translation_prompt(src_lang, trg_lang, source, prompt_version="codeblock")
        for source, src_lang, trg_lang in zip(src_list, src_langs, trg_langs)
    ]
    if not prompts:
        return {"translations": [], "responses": [], "thinking": []}
    if base_url:
        results = _run_remote(
            prompts, base_url=base_url, model_path=model_path, api_key=api_key,
            workers=workers, timeout=timeout, reasoning_effort=reasoning_effort,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, retry=retry,
        )
    else:
        llm = init_oss_model(model_path or "openai/gpt-oss-120b") if model is None else model
        encoding = load_encoding()
        results = _generate_with_retries(
            llm,
            _prepare_inputs(prompts, encoding, reasoning_effort),
            extract_translation_response,
            encoding,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry=retry,
        )
    return {
        "translations": [
            result["parsed"]["translation"] if result["parsed"] else None
            for result in results
        ],
        "responses": [result["response"] for result in results],
        "thinking": [result["thinking"] for result in results],
    }
