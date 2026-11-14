"""Memory APIs, whole-request vLLM submission, artifacts, and dependency boundaries."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data.candidate_error_workflow import annotate as save_annotations, read_annotation_stage
from data.candidate_error_workflow import predict as save_predictions
from inference.candidate_error_generation import run_requests
from inference.run_inst_candidate_error_prediction import predict as predict_inst
from inference.run_oss_candidate_error_prediction import predict as predict_oss
from inference.run_oss_candidate_error_annotation import annotate
from tests.test_candidate_error_pipeline import item, span
from utils.candidate_error_protocol import identity, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class CandidateRefactorTests(unittest.TestCase):
    def test_retained_prompt_text_matches_frozen_hashes(self):
        import hashlib
        from inference.candidate_error_prompts import build_prediction_prompt
        from inference.oss_candidate_error_prompts import build_annotation_prompt
        fixture = {'src_text': 'The board did not approve.', 'candidate': '董事会批准了。', 'src_lang': 'en', 'trg_lang': 'zh', 'ref_text': '董事会没有批准。', 'comment': 'test'}
        expected = {'annotation': 'a5102896466137ffb66d17d630ce2ad48c90ddff4ac3c8bdc617dd51b44fa639',
         'compact_analysis': 'fdf5ee632a768d73a1e5e6b0f28bd86f8e76630d1fc51b2fc8285a9c2daae0a1',
         'compact_direct': '4ffdc4f26486c0c3b0bd793c68d0828be19e6480de044eb8e7581d2ca5718b88',
         'compact_probability': '79ded8b354c2fcf42eb15dbf3d381047af0e674b11ae89e8772d151042c4e53f',
         'critical_audit': 'ed60cdf72085b731e3d0826de3a1f3bc74eb367acd12d31faa466d45e7b50d97',
         'critical_combined': 'aa9a4ccb84fd657a4a22f41593f391a76cdd63f13565e22296e233e5713ae4ac',
         'critical_combined_minor': '87a666cf1b9025d5b4b01aac9d8eeab4fd479768f5e6a50d667b324bb60db6d2',
         'critical_distilled': '2b85c3dffedf8bc27969a8cc1dd3af51f6441c798134cdd2c87c797541ce697b',
         'critical_frame': 'cbce38aa86870ef4f25cb61a7264be4c61af69a4364b54039ee0b3cfe5f0a7a3',
         'critical_micro': '84f4dd2d331a9c1dcef8babd153da4d85d9f6cce026ce4bbe4836629d586ad2d',
         'critical_minor': 'ebf5ed8512391e26525b1fca28835df5ece79820f4d056cc649f63088e5b3082',
         'critical_recall': '1d2ed37ef8f8375e9c5de3b8e6597bea8c71265bdf18f82cbc3efc465aeaba43',
         'critical_short': '9ac96eb7e4e9d834ea286e0ea9ace776a88cda65284bd7b85d513612f88b3a24',
         'critical_short_examples': 'd600abbf80b1bb2587ed773d384587a44fff5f749aba2efc42936cf1feda4929',
         'critical_trimmed': '8a5ec3337995d807c54a611fcfc63dcd9975cbfc1c4f830d72a51976ebc7cfd0',
         'legacy': '506467d054302fde7fb895f41311962c7a030dd2c234d8f1826c1d788037d97d',
         'legacy_analysis': '5d47a718fcc1a7cad7629ed31a4d4f493b28d3f257810b6eee7a2f1d0c64b332',
         'source_first': '9bd2890e91a324f7a8692bbc18fcb7954b0614d86c9729b7730df517ed72c79f'}
        for variant, checksum in expected.items():
            prompt = (build_annotation_prompt(fixture) if variant == 'annotation' else
                      build_prediction_prompt(fixture, variant,
                                              scoring='severity' if variant == 'legacy' else 'probability'))
            self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), checksum, variant)

    def test_inference_has_no_reverse_imports_or_file_processing(self):
        paths = list((ROOT / 'inference').glob('*candidate*.py'))
        self.assertEqual(len(list((ROOT / 'inference').glob('*candidate*prompts.py'))), 3)
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0], ('data', 'eval', 'scripts', 'pathlib'), path)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ('open', 'read_jsonl', 'write_json', 'write_jsonl'), path)
                if isinstance(node, ast.arg):
                    self.assertNotIn(node.arg, ('input_file', 'output_path', 'output_dir', 'batch_size',
                                               'dual', 'fixed', 'proposals', 'json_mode'), path)

    def test_memory_inputs_order_missing_and_empty(self):
        inputs = [dict(item(), candidate=None), dict(item(), row_id=1), dict(item(), row_id=2)]
        def generate(prompts, validators, **options):
            self.assertEqual(len(prompts), 2)
            return [dict(parsed=validator({'spans': []})) for validator in validators]
        for predict in (predict_inst, predict_oss):
            with patch('inference.hard_generation.generate', side_effect=generate), \
                 patch('builtins.open', side_effect=AssertionError('file IO in inference')):
                rows = predict(inputs, 'test', model=object())
                self.assertEqual(predict([], 'test'), [])
            self.assertEqual([r['id'] for r in rows], [identity(it) for it in inputs])
            self.assertEqual([r['status'] for r in rows], ['failed', 'ok', 'ok'])
            self.assertIsNone(rows[0]['result'])
            self.assertEqual(rows[1]['result']['spans'], [])

    def test_invalid_input_and_removed_options_fail_without_generation(self):
        for function in (predict_inst, predict_oss, annotate):
            for inputs in ('input.jsonl', {'candidate': 'text'}, [item(), item()]):
                with self.subTest(function=function, inputs=inputs), self.assertRaises((TypeError, ValueError)):
                    function(inputs, 'test')
            for option in ('dual', 'proposals', 'json_mode', 'batch_size', 'output_path'):
                with self.subTest(option=option), self.assertRaises(TypeError):
                    function([item()], 'test', **{option: True})

    def test_all_requests_reach_adapter_in_one_submission(self):
        inputs = [dict(item(), row_id=i) for i in range(97)]
        def generate(prompts, validators, **options):
            self.assertEqual(len(prompts), 97)
            return [dict(parsed=validator({'spans': []})) for validator in validators]
        for function in (predict_inst, predict_oss):
            with patch('inference.hard_generation.generate', side_effect=generate) as call:
                rows = function(inputs, 'test', model=object())
            self.assertEqual(len(rows), 97)
            self.assertEqual(call.call_count, 1)

    def test_actual_adapters_retry_all_and_only_failed_requests(self):
        from inference.hard_generation import generate
        from inference import run_oss_SQM, run_oss_diverse_mt, run_inst_mt
        prompts = [str(i) for i in range(65)]
        validators = [lambda value: value for _ in prompts]
        for backend in ('inst', 'oss'):
            calls = []
            def render_outputs(inputs, sampling_params):
                calls.append(list(inputs))
                # Two invalid first responses; only these must enter the second call.
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    text='bad' if len(calls) == 1 and int(value) in (1, 64) else '{"spans":[]}',
                    token_ids='bad' if len(calls) == 1 and int(value) in (1, 64) else '{"spans":[]}',
                    finish_reason='stop')]) for value in inputs]
            model = SimpleNamespace(generate=render_outputs)
            if backend == 'inst':
                model = SimpleNamespace(model=model)
            encoding = SimpleNamespace(stop_tokens_for_assistant_actions=lambda: [])
            with self.subTest(backend=backend), \
                 patch.dict(sys.modules, vllm=SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw))), \
                 patch('inference.run_inst_mt._render_prompt', side_effect=lambda engine, prompt, thinking: prompt), \
                 patch('inference.run_oss_SQM.load_encoding', return_value=encoding), \
                 patch('inference.run_oss_diverse_mt._prepare_inputs', side_effect=lambda prompts, *args: prompts), \
                 patch('inference.run_oss_diverse_mt._response_text', side_effect=lambda tokens, enc: (None, tokens)):
                rows = generate(prompts, validators, backend=backend, model_path='test', model=model, retry=1)
            self.assertEqual(calls, [prompts, ['1', '64']])
            self.assertTrue(all(r['parsed'] == {'spans': []} for r in rows))
            self.assertEqual([len(rows[i]['attempts']) for i in (0, 1, 64)], [1, 2, 2])

    def test_review_adjudication_preserves_stages_and_failures(self):
        calls = []
        def generate(prompts, validators, **options):
            stage = ['initial', 'independent', 'review', 'adjudication'][len(calls)]
            calls.append(stage)
            value = {'spans': [span()]} if stage == 'initial' else {'spans': []}
            return [dict(parsed=None if stage == 'adjudication' else validate(value), attempts=[])
                    for validate in validators]
        with patch('inference.hard_generation.generate', side_effect=generate):
            row = annotate([item()], 'test', model=object())[0]
        self.assertEqual(calls, ['initial', 'independent', 'review', 'adjudication'])
        self.assertTrue(row['disputed'])
        self.assertEqual(row['status'], 'failed')
        self.assertIsNone(row['result'])
        self.assertEqual(list(row['annotation_stages']), calls)

    def test_only_final_artifacts_are_saved_and_repeated_calls_regenerate(self):
        def generate(prompts, validators, **options):
            return [dict(parsed=validate({'spans': []}), attempts=[]) for validate in validators]
        with tempfile.TemporaryDirectory() as tmp, patch('inference.hard_generation.generate', side_effect=generate) as call:
            root = Path(tmp)
            save_annotations([item()], root / 'silver', 'test', model=object())
            self.assertEqual({p.name for p in (root / 'silver').iterdir()}, {'final.jsonl', 'summary.json'})
            initial = read_annotation_stage(root / 'silver', 'initial')
            self.assertEqual(initial[0]['result'], {'spans': []})
            self.assertEqual(read_annotation_stage(root / 'silver', 'adjudication'), [])
            save_annotations([], root / 'empty', 'test')
            self.assertEqual(read_annotation_stage(root / 'empty', 'initial'), [])
            path = root / 'predictions.jsonl'
            Path(str(path) + '.input_failures.json').write_text('[{"id":"stale"}]')
            save_predictions([dict(item(), candidate=None), dict(item(), row_id=1)], path, 'test', model=object())
            self.assertEqual([r['status'] for r in read_jsonl(path)], ['failed', 'ok'])
            before = call.call_count
            save_predictions([dict(item(), row_id=2)], path, 'test', model=object())
            self.assertEqual(call.call_count, before + 1)
            self.assertEqual(len(read_jsonl(path)), 1)
            self.assertFalse(list(root.glob('*.input_failures.json')))

    def test_historical_annotation_stage_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'final.jsonl').write_text('{"id":"x"}\n')
            (root / 'initial.jsonl').write_text('{"id":"x","status":"failed"}\n')
            self.assertEqual(read_annotation_stage(root, 'initial'), [{'id': 'x', 'status': 'failed'}])

    def test_generation_count_mismatch_is_rejected(self):
        with patch('inference.hard_generation.generate', return_value=[]), self.assertRaisesRegex(ValueError, 'count'):
            run_requests([item()], ['prompt'], model_path='test', model=object())


if __name__ == '__main__':
    unittest.main()
