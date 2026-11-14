"""OSS offline routing and development-only compression decisions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.run_oss_candidate_error_prediction import predict
from scripts.experiment_oss_candidate import BASE, SHORT, VARIANTS, choose, prompt
from tests.test_qwen_candidate_error import item, output


class OSSTransferTests(unittest.TestCase):
    def test_offline_oss_all_variants_ground_minor_and_never_call_http(self):
        for variant in VARIANTS:
            def generate(prompts, validators, **kwargs):
                self.assertEqual(kwargs['backend'], 'oss')
                self.assertEqual(kwargs['reasoning_effort'], 'medium')
                self.assertEqual(prompts, [prompt(item(), variant)])
                context = json.loads(prompts[0].rsplit('\nNow check this translation:\n', 1)[1])
                self.assertEqual(set(context), {'src_text','candidate','src_lang','trg_lang'})
                value = output(score=2)
                if variant != 'source_first':
                    value['spans'][0]['severity'] = 'minor'
                parsed = validators[0](value)
                value['spans'][0]['source_span'] = 'missing'
                self.assertIsNone(validators[0](value))
                return [{'parsed':parsed}]
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                with patch('inference.hard_generation.generate', side_effect=generate), patch(
                        'inference.candidate_error_generation.online_one', side_effect=AssertionError('HTTP in offline path')):
                    records = predict([item()], 'oss-model',
                        prompt_variant=variant, reasoning_effort='medium', model_factory=lambda:object())
                self.assertEqual(records[0]['status'], 'ok')
                self.assertEqual(records[0]['config']['base_url'], None)
                self.assertEqual(records[0]['result']['spans'][0]['confidence'], .2)

    def test_selection_rejects_ranking_regression_even_when_f1_improves(self):
        def stats(tokens, f1=.5, ap=.3, rho=.4):
            return dict(raw=dict(f1=f1, recall=.4), ap=ap,
                        reliability=dict(span_correctness_spearman=rho), instruction_tokens=tokens)
        summary = {BASE: stats(1800), **{n: stats(1000+i*100) for i,n in enumerate(SHORT)}}
        summary['critical_micro'] = stats(279, f1=.6, rho=.1)
        result = choose(summary)
        self.assertTrue(result['constraints_satisfied'])
        self.assertNotIn('critical_micro', result['eligible'])
        self.assertEqual(result['selected'], 'critical_trimmed')

    def test_data_cli_forwards_offline_oss_effort(self):
        from data.run_oss_candidate_error_prediction import main
        with patch('data.run_oss_candidate_error_prediction.load_items', return_value=[item()]), patch(
                'data.run_oss_candidate_error_prediction.predict') as call:
            main('input.jsonl', 'output.jsonl', 'oss-model',
                 prompt_variant='critical_micro', reasoning_effort='high')
        self.assertIsNone(call.call_args.kwargs['base_url'])
        self.assertEqual(call.call_args.kwargs['reasoning_effort'], 'high')
        self.assertNotIn('batch_size', call.call_args.kwargs)

    def test_dedicated_oss_entry_is_in_memory(self):
        with patch('inference.run_oss_candidate_error_prediction.predict_candidates', return_value=[]) as call:
            predict([item()], 'oss-model', prompt_variant='critical_trimmed')
        self.assertEqual(call.call_args.args, ([item()], 'oss-model'))
        self.assertEqual(call.call_args.kwargs['backend'], 'oss')
        self.assertEqual(call.call_args.kwargs['max_tokens'], 8192)


if __name__ == '__main__':
    unittest.main()
