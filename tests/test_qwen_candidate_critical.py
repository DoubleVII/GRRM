import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.candidate_error_prompts import build_critical_prompt, CRITICAL_VARIANTS as VARIANTS, MINOR_VARIANTS
from inference.run_inst_candidate_error_prediction import predict
from utils.candidate_error_response import validate_compact
from tests.test_qwen_candidate_error import item, output
from scripts.experiment_qwen_critical import major_gate
from eval.candidate_error_core import aggregate
from eval.run_candidate_error_eval import prepare_rows


class CriticalPromptTests(unittest.TestCase):
    def test_failed_judge_has_separate_budget_rescue_and_keeps_primary_failure(self):
        it = item()
        spans = validate_compact(output(), it)['spans']
        record = dict(id='x', item=it, status='ok', result={'spans': spans})
        failure = dict(status='failed', result=None)
        calls = []
        def requests(items, prompts, **kwargs):
            calls.append(kwargs['max_tokens'])
            if len(calls) == 1:
                return [failure]
            value = kwargs['validator_factory'](0)({'analysis': 'same source location', 'pairs': [[0, 0]]})
            return [dict(status='ok', result=value)]
        with tempfile.TemporaryDirectory() as tmp, patch('eval.run_candidate_error_eval.run_requests', side_effect=requests):
            rows, audit = prepare_rows([record], [record], tmp, model_path='oss', base_url='http://test',
                                       max_tokens=2048, rescue_max_tokens=8192)
        self.assertEqual(calls, [2048, 8192])
        self.assertEqual(failure['status'], 'failed')
        self.assertEqual(rows[0]['edges'], [[0, 0]])
        self.assertEqual((audit['primary_matching_failed'], audit['matching_rescued'], audit['matching_failed']), (1, 1, 0))

    def test_unsuccessful_rescue_still_contributes_zero_matches(self):
        it = item()
        record = dict(id='x', item=it, status='ok', result={'spans': validate_compact(output(), it)['spans']})
        with tempfile.TemporaryDirectory() as tmp, patch('eval.run_candidate_error_eval.run_requests',
                return_value=[dict(status='failed', result=None)]):
            rows, audit = prepare_rows([record], [record], tmp, model_path='oss', base_url='http://test',
                                       max_tokens=2048, rescue_max_tokens=8192)
        self.assertEqual(rows[0]['edges'], [])
        self.assertEqual(audit['matching_failed'], 1)
        self.assertEqual(audit['matching_rescued'], 0)

    def test_all_prompts_are_isolated_and_have_explicit_analysis(self):
        for variant in VARIANTS:
            prompt = build_critical_prompt(item(), variant)
            self.assertNotIn('SECRET_REF', prompt)
            self.assertNotIn('SECRET_COMMENT', prompt)
            context = json.loads(prompt.rsplit('\nNow check this translation:\n', 1)[1])
            self.assertEqual(set(context), {'src_text', 'candidate', 'src_lang', 'trg_lang'})
            self.assertIn('# Analysis', prompt)
            if 'combined' in variant or variant == 'critical_recall':
                self.assertIn('Do not impose', prompt)
            if variant in MINOR_VARIANTS:
                self.assertIn('BOTH major and minor', prompt)
                self.assertIn('score retains the SAME event', prompt)

    def test_optional_severity_preserves_minor_and_major_event_score(self):
        value = output(score=2)
        value['spans'][0]['severity'] = ' MINOR '
        normalized = validate_compact(value, item(), allow_minor=True)
        self.assertEqual(normalized['spans'][0]['severity'], 'minor')
        self.assertEqual(normalized['spans'][0]['confidence'], .2)
        self.assertEqual(validate_compact(value, item())['spans'][0]['severity'], 'major')
        for severity in (None, 'critical', 1):
            value['spans'][0]['severity'] = severity
            self.assertIsNone(validate_compact(value, item(), allow_minor=True))

    def test_minor_variant_uses_same_offline_parser_with_severity_validation(self):
        def fake_generate(prompts, validators, **kwargs):
            value = output(score=2)
            value['spans'][0]['severity'] = 'minor'
            self.assertEqual(kwargs['response_format'], 'candidate_relaxed')
            return [{'parsed': validators[0](value)}]
        with tempfile.TemporaryDirectory() as tmp:
            with patch('inference.hard_generation.generate', side_effect=fake_generate):
                records = predict([item()], 'model', prompt_variant='critical_minor',
                                  model_factory=lambda: object())
            self.assertEqual(records[0]['result']['spans'][0]['severity'], 'minor')

    def test_major_gate_preserves_edges_and_all_error_cost_is_separate(self):
        major = dict(source_span='a', source_start=0, source_end=1, severity='major', confidence=.9)
        minor = dict(source_span='b', source_start=1, source_end=2, severity='minor', confidence=.2)
        row = dict(id='x', item={'src_text': 'ab'}, predicted=[minor, major], gold=[major, minor],
                   edges=[[0, 1], [1, 0]], prediction_status='ok')
        raw = aggregate([row], .1, 'probability')
        any_error = aggregate([row], .1, 'probability', major=False)
        gated = aggregate(major_gate([row]), .1, 'probability')
        self.assertEqual((raw['tp'], raw['fp']), (1, 1))
        self.assertEqual((any_error['tp'], any_error['fp']), (2, 0))
        self.assertEqual((gated['tp'], gated['fp']), (1, 0))
        self.assertEqual(raw['triggers_per_100'], 200)
        self.assertEqual(gated['triggers_per_100'], 100)


if __name__ == '__main__':
    unittest.main()
