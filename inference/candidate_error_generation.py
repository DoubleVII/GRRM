"""Offline by default; bounded online experiment adapter shares prompts and validation."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.request

from utils.candidate_error_protocol import VERSION, digest, identity, validate
from utils.difficulty_response import parse_prediction_response


def online_one(prompt, validator, *, base_url, model_path, backend, temperature=0.1,
               top_p=1.0, max_tokens=4096, retry=2, reasoning_effort='medium',
               format_feedback=False, response_format='json'):
    if retry < 0:
        raise ValueError('retry must be nonnegative')
    attempts = []
    for attempt in range(retry + 1):
        body = dict(model=model_path, messages=[{'role': 'user', 'content': prompt}],
                    temperature=min(1., temperature + .1 * attempt), top_p=top_p, max_tokens=max_tokens)
        if attempt and format_feedback:
            if response_format == 'candidate_relaxed':
                body['messages'][0]['content'] += (
                    '\nThe previous output could not be parsed or grounded. Follow the requested schema. '
                    'Finish with valid JSON in a json code block. Copy source_span from the source; '
                    'include more context if the quote repeats. Use the requested score range.\n')
            else:
                body['messages'][0]['content'] += ('\nThe previous attempt failed strict validation. '+
                    ('Preserve # Step-by-step Analysis followed by # Final JSON and a single final json code fence; use valid JSON inside it. '
                     if response_format=='analysis_codeblock' else 'Return valid JSON; ')+
                    'apostrophes need no JSON escaping. Copy source quotes exactly. source_occurrence is ZERO based: '
                    'a quote appearing once always has source_occurrence:0. Do not invent, normalize or conjugate source text. '
                    '\n')
        if backend == 'oss':
            body['reasoning_effort'] = reasoning_effort
        else:
            body['chat_template_kwargs'] = {'enable_thinking': False}
        started = time.monotonic()
        record = dict(request=body, valid=False)
        parsed = None
        try:
            req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=600) as response:
                raw = json.load(response)
            message = raw['choices'][0]['message']
            answer = message.get('content') or ''
            parsed = validator(parse_prediction_response(answer,response_format))
            record.update(response=raw, valid=parsed is not None)
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
        record['seconds'] = time.monotonic() - started
        attempts.append(record)
        if parsed is not None:
            return dict(parsed=parsed, attempts=attempts)
    return dict(parsed=None, attempts=attempts)


def check_items(items):
    """Check the in-memory input contract before filtering invalid candidates."""
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise TypeError('items must be a list of dictionaries')
    keys = [identity(item) for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate candidate identities')


def run_requests(items, prompts, *, model_path, backend='inst', scoring='severity',
                 base_url=None, workers=8, validator_factory=None, model=None,
                 model_factory=None, temperature=0.1, top_p=1.0, max_tokens=4096,
                 retry=2, reasoning_effort='medium', vllm_kwargs=None,
                 format_feedback=False, response_format='json', enable_thinking=False,
                 think_tag='think'):
    """Generate all requests in memory; each offline attempt submits every pending request."""
    check_items(items)
    if not isinstance(prompts, list) or len(items) != len(prompts):
        raise ValueError('Request lengths differ or prompts is not a list')
    if backend not in ('inst', 'oss'):
        raise ValueError('Unknown backend')
    if retry < 0:
        raise ValueError('retry must be nonnegative')
    if reasoning_effort not in ('low', 'medium', 'high'):
        raise ValueError('reasoning_effort must be low, medium, or high')
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                   retry=retry, reasoning_effort=reasoning_effort,
                   vllm_kwargs=vllm_kwargs or {}, format_feedback=format_feedback,
                   response_format=response_format, enable_thinking=enable_thinking,
                   think_tag=think_tag)
    config = dict(version=VERSION, model_path=model_path, backend=backend,
                  scoring=scoring, base_url=base_url, options=options)
    if not items:
        return []
    # Validate trace metadata before loading a model or spending generation tokens.
    request_hashes = [digest([config, item, prompt]) for item, prompt in zip(items, prompts)]

    def validator(i):
        if validator_factory is not None:
            return validator_factory(i)
        return lambda value: validate(value, items[i], scoring=scoring)

    if base_url:
        endpoints = base_url if isinstance(base_url, (list, tuple)) else [base_url]
        if not endpoints or not all(isinstance(url, str) and url for url in endpoints):
            raise ValueError('Empty online endpoint')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(
                online_one, prompt, validator(i), base_url=endpoints[i % len(endpoints)],
                model_path=model_path, backend=backend, temperature=temperature,
                top_p=top_p, max_tokens=max_tokens, retry=retry,
                reasoning_effort=reasoning_effort, format_feedback=format_feedback,
                response_format=response_format) for i, prompt in enumerate(prompts)]
            generations = [future.result() for future in futures]
    else:
        from inference.hard_generation import generate
        if model is None and model_factory is not None:
            model = model_factory()
        generations = generate(
            prompts, [validator(i) for i in range(len(items))], backend=backend,
            model_path=model_path, model=model, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, retry=retry, reasoning_effort=reasoning_effort,
            vllm_kwargs=vllm_kwargs or {}, response_format=response_format,
            enable_thinking=enable_thinking, think_tag=think_tag)
    if len(generations) != len(items):
        raise ValueError('Generation count mismatch')
    return [dict(id=identity(item), item=item, config=config, prompt=prompt,
                 request_hash=request_hash,
                 status='ok' if generation.get('parsed') is not None else 'failed',
                 result=generation.get('parsed'), generation=generation)
            for item, prompt, generation, request_hash in zip(items, prompts, generations, request_hashes)]
