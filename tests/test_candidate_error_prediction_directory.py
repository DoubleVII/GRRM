"""INST and SFT directory prediction preserves identities and backend options."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from data import merge_run_candidate_error_prediction as sft
from data import merge_run_inst_candidate_error_prediction as inst
from data import run_inst_candidate_error_prediction as single_inst
from data import run_oss_candidate_error_prediction as single_oss
from utils.candidate_error_protocol import digest, identity, read_jsonl


class PredictionDirectoryTests(unittest.TestCase):
    def test_single_file_entries_call_inference_directly_and_overwrite_output(self):
        for module, backend in ((single_inst, 'inst'), (single_oss, 'oss')):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                self.assertEqual(module.predict.__module__,
                                 f'inference.run_{backend}_candidate_error_prediction')
                root = Path(directory)
                self.inputs(root / 'input')
                output = root / 'predictions.jsonl'
                def infer(items, model_path, **options):
                    self.assertEqual(model_path, 'test-model')
                    self.assertNotIn('backend', options)
                    self.assertEqual(options['max_tokens'], 8192)
                    if backend == 'oss':
                        self.assertEqual(options['reasoning_effort'], 'high')
                    else:
                        self.assertIs(options['enable_thinking'], True)
                        self.assertNotIn('reasoning_effort', options)
                    self.assertTrue(all('ref_text' not in item and 'comment' not in item
                                        for item in items))
                    return [dict(id=identity(item), item=item,
                                 status='failed' if item['candidate'] is None else 'ok')
                            for item in items]
                with patch.object(module, 'predict', side_effect=infer) as call:
                    rows = module.main(root / 'input/a.parquet', output, 'test-model',
                                       **(dict(reasoning_effort='high') if backend == 'oss' else dict(enable_thinking=True)))
                    self.assertEqual(read_jsonl(output), rows)
                    self.assertEqual(len(rows), 2)
                    rows = module.main(root / 'input/nested/b.parquet', output, 'test-model',
                                       **(dict(reasoning_effort='high') if backend == 'oss' else dict(enable_thinking=True)))
                    self.assertEqual(read_jsonl(output), rows)
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['status'], 'failed')
                    self.assertEqual(call.call_count, 2)

    def inputs(self, root, collision=False):
        for filename, dataset, candidates in (
            ('a.parquet', 'a', ['one', 'two']),
            ('nested/b.parquet', 'a' if collision else 'b', [None]),
        ):
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                dict(data_id=dataset, row_id=0, candidate_index=i, candidate_model='inst',
                     src_text='source', candidate=candidate, src_lang='en', trg_lang='zh',
                     ref_text='reference', comment='reference evidence')
                for i, candidate in enumerate(candidates)
            ]).to_parquet(path, index=False)

    def infer(self, items, model_path, **options):
        self.assertEqual(model_path, 'test-model')
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertNotIn('ref_text', item)
            self.assertNotIn('comment', item)
        return [dict(id=identity(item), item=item,
                     status='failed' if item['candidate'] is None else 'ok', result=None)
                for item in reversed(items)]

    def test_defaults_single_call_split_and_summary(self):
        for module in (inst, sft):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.inputs(root / 'input')
                with patch.object(module, 'predict', side_effect=self.infer) as call:
                    module.main(root / 'input', root / 'out', 'test-model')
                call.assert_called_once()
                options = call.call_args.kwargs
                self.assertEqual(options['max_tokens'], 8192)
                self.assertEqual(options['scoring'], 'probability' if module is inst else 'score10')
                self.assertEqual(options['prompt_variant'], 'auto' if module is inst else 'critical_micro_fit')
                self.assertIs(options['enable_thinking'], False)
                if module is sft:
                    self.assertEqual(options['think_tag'], 'think')
                else:
                    self.assertNotIn('think_tag', options)
                self.assertEqual(options['top_p'], 1.0)
                self.assertNotIn('reasoning_effort', options)
                rows = read_jsonl(root / 'out/a/predictions.jsonl')
                self.assertEqual([row['id'] for row in rows], ['a:0:inst:0', 'a:0:inst:1'])
                missing = read_jsonl(root / 'out/nested/b/predictions.jsonl')
                self.assertEqual(missing[0]['id'], 'b:0:inst:0')
                self.assertEqual(missing[0]['status'], 'failed')
                summary = json.loads((root / 'out/merged_summary.json').read_text())
                self.assertEqual((summary['total'], summary['failed']), (3, 1))
                self.assertEqual(summary['input_digest'], digest(call.call_args.args[0]))
                self.assertEqual(summary['options'], {
                    key: value for key, value in options.items()
                    if key not in ('scoring', 'prompt_variant')})
                self.assertEqual([f['input_file'] for f in summary['files']],
                                 ['a.parquet', 'nested/b.parquet'])
                file_summary = json.loads((root / 'out/nested/b/summary.json').read_text())
                self.assertEqual((file_summary['total'], file_summary['failed']), (1, 1))

    def test_explicit_options_and_correct_inference_interfaces(self):
        self.assertEqual(inst.predict.__module__, 'inference.run_inst_candidate_error_prediction')
        self.assertEqual(sft.predict.__module__, 'inference.run_candidate_error_prediction')
        common = dict(scoring='score10', prompt_variant='critical_micro_fit',
                      base_url='http://localhost:8000/v1', workers=3, temperature=0.7,
                      max_tokens=1234, retry=4, vllm_kwargs={'tensor_parallel_size': 2})
        for module in (inst, sft):
            extra = dict(top_p=0.8, enable_thinking='True')
            if module is sft:
                extra['think_tag'] = 'thinking'
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.inputs(root / 'input')
                with patch.object(module, 'predict', side_effect=self.infer) as call:
                    module.main(root / 'input', root / 'out', 'test-model', **common, **extra)
                expected = dict(common, **extra)
                expected['enable_thinking'] = True
                self.assertEqual(call.call_args.kwargs, expected)

    def test_invalid_inputs_fail_before_inference(self):
        for module in (inst, sft):
            for case in ('missing', 'empty', 'same', 'nested', 'collision', 'metadata'):
                with self.subTest(module=module.__name__, case=case), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source, target = root / 'input', root / 'out'
                    if case == 'empty':
                        source.mkdir()
                    elif case != 'missing':
                        self.inputs(source, collision=case == 'collision')
                    if case == 'same':
                        target = source
                    elif case == 'nested':
                        target = source / 'out'
                    options = {'vllm_kwargs': {'invalid': object()}} if case == 'metadata' else {}
                    with patch.object(module, 'predict') as call:
                        with self.assertRaises((ValueError, TypeError)):
                            module.main(source, target, 'test-model', **options)
                    call.assert_not_called()
                    self.assertFalse((root / 'out').exists())

    def test_invalid_results_write_no_artifacts(self):
        for module in (inst, sft):
            for case in ('missing', 'duplicate', 'unexpected'):
                with self.subTest(module=module.__name__, case=case), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.inputs(root / 'input')
                    def infer(items, model_path, **options):
                        rows = self.infer(items, model_path, **options)
                        if case == 'missing':
                            return rows[:-1]
                        if case == 'duplicate':
                            return rows + rows[:1]
                        return rows + [dict(id='unexpected', status='ok')]
                    with patch.object(module, 'predict', side_effect=infer):
                        with self.assertRaisesRegex(ValueError, 'Result identities'):
                            module.main(root / 'input', root / 'out', 'test-model')
                    self.assertFalse((root / 'out').exists())

    def test_invalid_think_tag_fails_before_inference(self):
        with patch.object(sft, 'predict') as call:
            with self.assertRaisesRegex(ValueError, 'think_tag'):
                sft.main('unused', 'unused', 'test-model', think_tag='<think>')
        call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
