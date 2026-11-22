"""Reference-assisted silver annotation with independent inspection and adjudication."""
from inference.candidate_error_generation import check_items, run_requests
from inference.candidate_error_prompts import build_prompt
from inference.oss_candidate_error_prompts import build_annotation_prompt
from utils.candidate_error_protocol import digest, identity
from utils.candidate_error_response import validate_annotation


def annotate(items, model_path, *, review=True, prompt_variant='legacy',
             workflow='review', samples=3, min_successful_samples=2, model=None,
             temperature=None, top_p=1.0, max_tokens=4096, retry=2,
             reasoning_effort='medium', vllm_kwargs=None, base_url=None, workers=8,
             format_feedback=False):
    check_items(items)
    options = dict(top_p=top_p, max_tokens=max_tokens,
                   retry=retry, reasoning_effort=reasoning_effort, vllm_kwargs=vllm_kwargs,
                   base_url=base_url, workers=workers, format_feedback=format_feedback)
    # Preserve the generator's default temperature and independent-stage override
    # when the caller does not explicitly choose a temperature.
    if temperature is not None:
        options['temperature'] = temperature
    if retry < 0:
        raise ValueError('retry must be nonnegative')
    if reasoning_effort not in ('low', 'medium', 'high'):
        raise ValueError('reasoning_effort must be low, medium, or high')
    original = items
    history = {}
    if prompt_variant not in ('legacy', 'critical') or workflow not in ('review', 'synthesis'):
        raise ValueError('Unknown annotation prompt/workflow')
    if (type(samples) is not int or samples < 2 or type(min_successful_samples) is not int
            or not 1 <= min_successful_samples <= samples):
        raise ValueError('Require samples >= 2 and 1 <= min_successful_samples <= samples')
    if workflow == 'synthesis' and not review:
        raise ValueError('synthesis requires review=True; use review=False for single annotation')
    if workflow == 'synthesis' and prompt_variant == 'legacy':
        raise ValueError('Use prompt_variant=critical for synthesis')

    def prompt(item, *, annotations=None, adjudication=None, sample=0):
        if prompt_variant == 'critical':
            return build_annotation_prompt(item, annotations=annotations,
                                           adjudication=adjudication, sample=sample)
        if annotations is not None and len(annotations) != 2:
            raise ValueError('Legacy synthesis supports exactly two samples; use critical')
        return build_prompt(item, annotation=True,
                            review=dict(initial=annotations[0], independent=annotations[1]) if annotations else None) + (
            '\nADJUDICATION: Resolve substantive disagreements conservatively from source/candidate evidence; inspect for missed errors.\n' +
            'Previous audit: ' + str(adjudication) if adjudication is not None else '')
    missing = [dict(id=identity(item), item=item, status='failed', result=None,
                    silver=True, error='Missing candidate translation')
               for item in items if not isinstance(item.get('candidate'), str)]
    items = [item for item in items if isinstance(item.get('candidate'), str)]

    def finish(rows, **metadata):
        by_id = {row['id']: dict(row, silver=True, annotation_prompt=prompt_variant,
                                annotation_stages={name: values[row['id']] for name, values in history.items()
                                                   if row['id'] in values}, **metadata)
                 for row in rows + missing}
        return [by_id[identity(item)] for item in original]

    def stage(name, selected, prompts, **changes):
        nonlocal model
        if selected and not base_url and model is None:
            # Reject unserializable trace metadata before loading the shared model.
            digest([selected, prompts, options, changes])
            from inference.run_oss_SQM import init_oss_model
            model = init_oss_model(model_path, **(vllm_kwargs or {}))
        critical = dict(scoring='probability', validator_factory=lambda i:
                        lambda v: validate_annotation(v, selected[i])) if prompt_variant == 'critical' else {}
        rows = run_requests(selected, prompts, model_path=model_path,
                            backend='oss', model=model,
                            **critical, **dict(options, **changes))
        history[name] = {row['id']: row for row in rows}
        return rows
    initial = stage('initial', items, [prompt(x) for x in items])
    if not review:
        return finish(initial, annotation_method='single')
    independent = stage('independent', items, [prompt(x, sample=1) for x in items],
                        temperature=max(0.3, options.get('temperature', 0.1)))
    if workflow == 'synthesis':
        sampled = [initial, independent]
        for j in range(2, samples):
            sampled.append(stage(f'sample_{j}', items, [prompt(x, sample=j) for x in items],
                                 temperature=max(0.3, options.get('temperature', 0.1))))
        annotations = [[values[i]['result'] for values in sampled if values[i]['status'] == 'ok']
                     for i in range(len(items))]
        eligible = [i for i, values in enumerate(annotations) if len(values) >= min_successful_samples]
        synthesized = stage('synthesis', [items[i] for i in eligible],
                            [prompt(items[i], annotations=annotations[i]) for i in eligible])
        by_id = {r['id']: r for r in synthesized}
        final = [dict(by_id.get(row['id'], dict(row, status='failed', result=None,
                      error='Insufficient successful independent samples')),
                      successful_samples=len(annotations[i]), requested_samples=samples)
                 for i, row in enumerate(initial)]
        return finish(final, annotation_method='independent_sampling_synthesis')
    eligible = [i for i in range(len(items)) if initial[i]['status'] == independent[i]['status'] == 'ok']
    reviewed = stage('review', [items[i] for i in eligible],
                     [prompt(items[i], annotations=[initial[i]['result'], independent[i]['result']]) for i in eligible])
    reviews = {r['id']: r for r in reviewed}

    # Changes in position/severity/type are substantive; explain-only changes are not.
    def signature(row):
        return sorted((s['source_start'] if s['source_start'] is not None else -1,
                       s['source_end'] if s['source_end'] is not None else -1,
                       s['severity'], s['error_type'], s.get('impact', 0))
                      for s in (row['result'] or {}).get('spans', []))
    disputed = [i for i in eligible if signature(initial[i]) != signature(independent[i])]
    adjudicated = stage('adjudication', [items[i] for i in disputed],
                        [prompt(items[i], annotations=[initial[i]['result'], independent[i]['result']],
                                adjudication=reviews[initial[i]['id']]['result']) for i in disputed])
    adjudications = {r['id']: r for r in adjudicated}
    final = []
    for i, row in enumerate(initial):
        chosen = adjudications.get(row['id'], reviews.get(row['id']))
        if chosen is None:
            chosen = dict(row, status='failed', result=None)
        final.append(dict(chosen, silver=True, disputed=i in disputed,
                          annotation_method='independent_review_adjudication'))
    return finish(final)
