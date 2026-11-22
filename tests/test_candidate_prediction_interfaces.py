"""Explicit prediction parameters and model-specific option boundaries."""
import inspect
import unittest
from unittest.mock import patch

from inference.run_candidate_error_prediction import predict as sft
from inference.run_inst_candidate_error_prediction import predict as inst
from inference.run_oss_candidate_error_prediction import predict as oss
from inference.candidate_error_prediction import predict_candidates
from tests.test_qwen_candidate_error import item


class PredictionInterfaceTests(unittest.TestCase):
    def test_signatures_are_explicit_and_backend_specific(self):
        for entry in (sft, inst, oss, predict_candidates):
            signature = inspect.signature(entry)
            self.assertFalse(any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()))
            for name in ('temperature', 'top_p', 'max_tokens', 'retry', 'model',
                         'vllm_kwargs', 'base_url', 'workers', 'format_feedback'):
                self.assertIn(name, signature.parameters)
        for entry in (sft, inst):
            self.assertIn('enable_thinking', inspect.signature(entry).parameters)
            with self.assertRaises(TypeError):
                entry([], 'test', reasoning_effort='high')
        self.assertIn('think_tag', inspect.signature(sft).parameters)
        self.assertNotIn('think_tag', inspect.signature(inst).parameters)
        with self.assertRaises(TypeError):
            inst([], 'test', think_tag='thinking')
        for option in ('enable_thinking', 'think_tag', 'backend', 'validator_factory', 'response_format'):
            with self.assertRaises(TypeError):
                oss([], 'test', **{option: 'unsupported'})

    def test_options_reach_generation_and_parsing_stays_internal(self):
        for entry in (sft, inst, oss):
            specific = dict(reasoning_effort='high') if entry is oss else dict(
                enable_thinking='True')
            if entry is sft:
                specific['think_tag'] = 'thinking'
            model = object()
            with patch('inference.hard_generation.generate', return_value=[dict(parsed={'spans': []})]) as call:
                rows = entry([item()], 'test', temperature=.3, top_p=.8, max_tokens=123,
                             retry=4, model=model, **specific)
            options = call.call_args.kwargs
            self.assertEqual((options['temperature'], options['top_p'], options['max_tokens'], options['retry']),
                             (.3, .8, 123, 4))
            self.assertIs(options['model'], model)
            self.assertEqual(options['response_format'], 'candidate_relaxed')
            self.assertEqual(options['backend'], 'oss' if entry is oss else 'inst')
            if entry is oss:
                self.assertEqual(options['reasoning_effort'], 'high')
            else:
                self.assertIs(options['enable_thinking'], True)
                self.assertEqual(options['think_tag'], 'thinking' if entry is sft else 'think')
            self.assertEqual(rows[0]['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
