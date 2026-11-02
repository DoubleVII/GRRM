"""Run both OSS entry points against the frozen Qwen benchmark candidates/golden."""
import argparse
import json
from pathlib import Path
from shutil import copy2
from unittest.mock import patch

from scripts.benchmark_hard_pipeline import DATASETS, OSS, Recorder, dump, read_stage
from inference.run_oss_hard_prediction import main as prediction_main
from eval.run_oss_hard_eval import main as evaluation_main
from eval.hard_eval_core import summarize
from utils.hard_protocol import align, artifact_path, output_paths


def main(input_dir, output_dir):
    source, output = Path(input_dir), Path(output_dir)
    for mode in ('direct', 'qe'):
        output_paths(DATASETS, output, 'predictions', mode)
        output_paths(DATASETS, output, 'evaluation', mode)
    output.mkdir(parents=True, exist_ok=True)
    for did in DATASETS:
        for stage in ('candidates', 'golden'):
            copy2(artifact_path(source, did, stage), artifact_path(output, did, stage))
    kwargs = dict(gpu_memory_utilization=0.85, max_model_len=16384,
                  max_num_seqs=32, seed=20261102, safetensors_load_strategy='eager')
    options = dict(model_path=OSS, temperature=0.0, top_p=1.0, max_tokens=8192,
                   retry=3, reasoning_effort='medium', vllm_kwargs=kwargs)
    dump(output / 'metadata.json', dict(input_dir=str(source), model=OSS, **{
        key: value for key, value in options.items() if key != 'model_path'},
        comparison='Same frozen Qwen translations, golden, prompts and validation; predictor changes to OSS.'))
    from inference.run_oss_SQM import init_oss_model
    recorder = Recorder(init_oss_model(OSS, **kwargs), output, 'oss')
    # Exercise the actual CLI main functions while loading the shared engine once.
    with patch('inference.run_oss_SQM.init_oss_model', return_value=recorder):
        for mode in ('direct', 'qe'):
            recorder.start(f'{mode}.round0')
            prediction_main(DATASETS, str(source), str(output), mode=mode, **options)
            recorder.start(f'{mode}.matching')
            evaluation_main(DATASETS, str(output), str(source), str(output), mode=mode, **options)
            items = read_stage(output, 'evaluation', mode)
            summary = dict(overall=summarize(items), by_dataset={
                did: summarize([x for x in items if x['data_id'] == did]) for did in DATASETS})
            dump(output / f'{mode}.summary.json', summary)
    # Compare the exact common successful item sets, excluding coverage differences.
    comparison = {}
    for mode in ('direct', 'qe'):
        old = read_stage(source, 'evaluation', mode)
        new = read_stage(output, 'evaluation', mode)
        common = {(x['data_id'], x['row_id']) for x, y in align(old, new)
                  if x['status'] == y['status'] == 'ok'}
        comparison[mode] = dict(
            qwen=summarize(old), oss=summarize(new), common_items=len(common),
            qwen_common=summarize([x for x in old if (x['data_id'], x['row_id']) in common]),
            oss_common=summarize([x for x in new if (x['data_id'], x['row_id']) in common]))
    dump(output / 'comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='results/hard_pipeline_20261102/relaxed')
    parser.add_argument('--output-dir', default='results/hard_pipeline_20261102/oss_baseline')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
