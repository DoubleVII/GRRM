"""Prompt-only A/B experiments that leave production prompts intact."""
import json
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_hard_pipeline import (QWEN, Recorder, dump, read_stage, save_stage)
from data.run_inst_hard_candidates import make_candidates
from inference.run_inst_hard_prediction import predict
from inference.hard_prompts import build_prompt
from inference.inst_mt_prompts import build_translation_prompt

COPY_RULE = '''
Before returning the JSON, verify each source_span is an EXACT, contiguous substring of src_text. Preserve original case, punctuation, spelling, spaces and line breaks; do not normalize, correct, paraphrase, translate, or join separate substrings. Prefer the shortest source quote that identifies the issue. Put interpretation and corrections only in explanation. If a quote does not occur exactly, replace it with a shorter exact quote. Use valid JSON escaping for quotes and newlines. Return the JSON object only.'''
CANDIDATE_RULE = '''
Output reminder: After # Final Translation, put the complete translation inside a triple-backtick code block. Write the opening ``` on its own line, then the translation, then the closing ``` on its own line. The closing ``` must be the last line of your response. Do not use code blocks in the analysis.'''


def candidate_prompt(*args, **kwargs):
    return build_translation_prompt(*args, **kwargs) + CANDIDATE_RULE


def prediction_prompt(item, mode):
    return build_prompt(item, mode) + COPY_RULE


def main():
    from inference.run_inst_mt import init_inst_model
    root = Path('results/hard_pipeline_20261102')
    output = root / 'prompt_variants'
    output.mkdir(exist_ok=True)
    dump(output / 'prompt_changes.json', dict(candidate_append=CANDIDATE_RULE, prediction_append=COPY_RULE))
    engine = init_inst_model(QWEN, gpu_memory_utilization=0.85, max_model_len=16384,
                             max_num_seqs=32, seed=20261102)
    recorder = Recorder(engine.model, output, 'inst')
    engine.model = recorder
    options = dict(temperature=0.9, top_p=1.0, max_tokens=8192, retry=3)
    recorder.start('candidates')
    items = json.loads((root / 'selection.json').read_text())
    with patch('inference.run_inst_mt.build_translation_prompt', candidate_prompt):
        revised = make_candidates(items, QWEN, model=engine, **options)
    save_stage(output / 'revised_candidates', 'candidates', revised)
    # Freeze the SAME baseline candidates for the difficulty-prompt comparison.
    candidates = read_stage(root, 'candidates')
    save_stage(output, 'candidates', candidates)
    for mode in ['direct', 'qe']:
        recorder.start(f'{mode}.round0')
        with patch('inference.run_inst_hard_prediction.build_prompt', prediction_prompt):
            predictions = predict(candidates, mode, QWEN, model=engine, **options)
        save_stage(output, 'predictions', predictions, mode)


if __name__ == '__main__':
    main()
