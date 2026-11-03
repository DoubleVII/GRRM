"""Real-model diagnostic for the difficulty pipeline; preserves every generation attempt."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

from data.run_inst_hard_candidates import load_items, make_candidates
from inference.run_inst_hard_prediction import predict
from eval.hard_eval_core import annotate, evaluate
from utils.hard_protocol import save, artifact_path

QWEN = '/home/nfs06/yangs/LLM/Qwen/Qwen3-4B-Instruct-2507'
OSS = '/home/zfs01/yangs/LLM/openai/gpt-oss-120b'
DATASETS = ['seedx_challenge_zhen', 'seedx_challenge_enzh', 'wmt23_zhen',
            'wmt24_enzh', 'wmt23_ja_en', 'wmt23_de_en']


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class Recorder:
    def __init__(self, model, root, backend):
        self.model, self.root, self.backend = model, root, backend
        self.stage = 'init'
        self.keys = {}
        self.calls = []
        self.encoding = None
        if backend == 'oss':
            from inference.run_oss_SQM import load_encoding
            self.encoding = load_encoding()

    def generate(self, prompts, sampling_params):
        started = time.monotonic()
        outputs = self.model.generate(prompts, sampling_params=sampling_params)
        records = []
        for prompt, output in zip(prompts, outputs):
            key = json.dumps(prompt, ensure_ascii=False)
            if key not in self.keys:
                self.keys[key] = len(self.keys)
            out = output.outputs[0]
            if self.backend == 'oss':
                from inference.run_oss_diverse_mt import _response_text
                thinking, response = _response_text(out.token_ids, self.encoding)
            else:
                from utils.helpers import split_thinking
                thinking, response = split_thinking(out.text)
            records.append(dict(index=self.keys[key], raw_output=out.text,
                                response=response, thinking=thinking,
                                output_tokens=len(out.token_ids),
                                finish_reason=out.finish_reason))
        self.calls.append(dict(stage=self.stage, batch=len(prompts),
                               seconds=time.monotonic()-started, records=records))
        dump(self.root / f'{self.backend}.attempts.json', self.calls)
        print(f'STAGE {self.stage}: batch={len(prompts)} seconds={self.calls[-1]["seconds"]:.1f}', flush=True)
        return outputs

    def start(self, stage):
        self.stage, self.keys = stage, {}
        print(f'START {stage}', flush=True)


def save_stage(root, stage, items, mode=None, **extra):
    for did in DATASETS:
        selected = [x for x in items if x['data_id'] == did]
        dataset_extra = dict(extra)
        if stage == 'evaluation':
            from eval.hard_eval_core import summarize
            dataset_summary = summarize(selected)
            dataset_extra['summary'] = dict(overall=dataset_summary,
                                             by_dataset={did: dataset_summary})
        save(artifact_path(root, did, stage, mode), stage,
             dict(mode=mode, benchmark=True, seed=20261102,
                  model_path=QWEN if stage in {'candidates', 'predictions'} else OSS,
                  temperature=0.9, max_tokens=8192, retry=3), selected, **dataset_extra)


def read_stage(root, stage, mode=None):
    return [item for did in DATASETS
            for item in json.loads(artifact_path(root, did, stage, mode).read_text())['items']]


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    kwargs = dict(gpu_memory_utilization=0.85, max_model_len=16384,
                  max_num_seqs=32, seed=20261102)
    options = dict(temperature=0.9, top_p=1.0, max_tokens=8192, retry=3)
    if args.phase == 'qwen':
        import random
        from inference.run_inst_mt import init_inst_model
        items = []
        for did in DATASETS:
            all_items = load_items(did)
            longest = sorted(all_items, key=lambda x: len(x['src_text']), reverse=True)[:4]
            remaining = [x for x in all_items if x['row_id'] not in {y['row_id'] for y in longest}]
            selected = random.Random(20261102).sample(remaining, 16) + longest
            items.extend(sorted(selected, key=lambda x: x['row_id']))
        dump(root / 'selection.json', items)
        engine = init_inst_model(QWEN, **kwargs)
        recorder = Recorder(engine.model, root, 'inst')
        engine.model = recorder
        recorder.start('candidates')
        candidates = make_candidates(items, QWEN, model=engine, **options)
        save_stage(root, 'candidates', candidates)
        for round_id in range(2):
            folder = root if round_id == 0 else root / f'repeat{round_id}'
            for mode in ['direct', 'qe']:
                recorder.start(f'{mode}.round{round_id}')
                predictions = predict(candidates, mode, QWEN, model=engine, **options)
                save_stage(folder, 'predictions', predictions, mode)
    else:
        from inference.run_oss_SQM import init_oss_model
        recorder = Recorder(init_oss_model(OSS, **kwargs), root, 'oss')
        candidates = read_stage(root, 'candidates')
        recorder.start('golden')
        golden = annotate(candidates, OSS, model=recorder, **options)
        save_stage(root, 'golden', golden)
        for mode in ['direct', 'qe']:
            recorder.start(f'{mode}.matching')
            predictions = read_stage(root, 'predictions', mode)
            items, summary = evaluate(predictions, golden, OSS, model=recorder, **options)
            save_stage(root, 'evaluation', items, mode, summary=summary)
            dump(root / f'{mode}.summary.json', summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        variants = root / 'prompt_variants'
        if (variants / f'{DATASETS[-1]}.qe.predictions.json').exists():
            variant_recorder = Recorder(recorder.model, variants, 'oss')
            save_stage(variants, 'golden', golden)
            for mode in ['direct', 'qe']:
                variant_recorder.start(f'{mode}.matching')
                predictions = read_stage(variants, 'predictions', mode)
                items, summary = evaluate(predictions, golden, OSS, model=variant_recorder, **options)
                save_stage(variants, 'evaluation', items, mode, summary=summary)
                dump(variants / f'{mode}.summary.json', summary)
                print('VARIANT ' + json.dumps(summary, ensure_ascii=False), flush=True)
        if any(x['status'] == 'failed' and x['candidate'] for x in golden):
            from inference.hard_prompts import golden_prompt
            from scripts.benchmark_hard_prompt_variants import COPY_RULE
            folder = root / 'golden_prompt_variant'
            folder.mkdir(exist_ok=True)
            variant_recorder = Recorder(recorder.model, folder, 'oss')
            variant_recorder.start('golden')
            rule = COPY_RULE + (' Every span must include source_span, explanation, '
                    'translation_span and error_type. For omission, translation_span is "". '
                    'For other errors, translation_span must be a nonempty EXACT contiguous '
                    'substring of candidate, with original case, spelling, spaces and punctuation. '
                    'Never quote from ref_text as translation_span. ')
            dump(folder / 'prompt_changes.json', dict(golden_append=rule))
            with patch('eval.hard_eval_core.golden_prompt', lambda x: golden_prompt(x) + rule):
                revised = annotate(candidates, OSS, model=variant_recorder, **options)
            save_stage(folder, 'golden', revised)
            save_stage(folder, 'candidates', candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['qwen', 'oss'])
    parser.add_argument('--output', default='results/hard_pipeline_20261102')
    run(parser.parse_args())
