"""Annotation-specific evidence, ordinal severity, failures and offline routing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.oss_candidate_error_prompts import build_annotation_prompt
from utils.candidate_error_response import validate_annotation
from inference.run_oss_candidate_error_annotation import annotate
from utils.candidate_error_protocol import identity
from tests.test_candidate_error_pipeline import item, span


class AnnotationTests(unittest.TestCase):
    def test_context_excludes_predictions_and_stale_annotations(self):
        prompt = build_annotation_prompt(dict(item(), prediction='SECRET_PREDICTION'))
        context = json.loads(prompt.split('CONTEXT_JSON:\n')[1])
        self.assertIn('SECRET_REFERENCE', prompt)
        self.assertIn('SECRET_COMMENT', prompt)
        self.assertNotIn('SECRET_ANNOTATION', prompt)
        self.assertNotIn('SECRET_PREDICTION', prompt)
        self.assertNotIn('independent_annotations', context)

    def test_impact_is_ordinal_and_separate_from_confidence(self):
        for impact in range(1,5):
            s = dict(span(), impact=impact, severity='major' if impact>=3 else 'minor', confidence=.4)
            self.assertEqual(validate_annotation({'spans':[s]}, item())['spans'][0]['impact'], impact)
            self.assertIsNone(validate_annotation({'spans':[dict(s, severity='minor' if impact>=3 else 'major')]}, item()))
        for value in (True, 0, 5, 2.5, None):
            self.assertIsNone(validate_annotation({'spans':[dict(span(),impact=value)]},item()))

    def test_grounding_and_confidence_remain_strict(self):
        for update in (dict(source_span='invented'),dict(confidence=float('nan')),dict(confidence=2)):
            self.assertIsNone(validate_annotation({'spans':[dict(span(), impact=3, **update)]}, item()))
        self.assertEqual(validate_annotation({'spans':[]},item()),{'spans':[]})

    def test_severity_is_derived_without_redundant_model_field(self):
        for impact in range(1,5):
            raw=dict(span(), impact=impact)
            del raw['severity']
            parsed=validate_annotation({'spans':[raw]},item())
            self.assertEqual(parsed['spans'][0]['severity'], 'major' if impact>=3 else 'minor')
        context=build_annotation_prompt(item()).split('SCHEMA: ')[1].split('\nCONTEXT_JSON:')[0]
        self.assertNotIn('severity', json.loads(context)['spans'][0])

    def test_synthesis_recovers_partial_failure_without_empty_imputation(self):
        calls = []
        def fake(items, prompts, **options):
            name = ['initial', 'independent', 'sample_2', 'synthesis'][len(calls)]
            calls.append((name, prompts))
            if name == 'synthesis':
                context = json.loads(prompts[0].split('CONTEXT_JSON:\n')[1])
                self.assertEqual(context['independent_annotations'], [{'spans': []}] * 2)
            return [dict(id=identity(it), item=it, status='failed' if name == 'initial' else 'ok',
                         result=None if name == 'initial' else {'spans': []}) for it in items]
        with patch('inference.run_oss_candidate_error_annotation.run_requests', side_effect=fake):
            rows = annotate([item()], 'unused', prompt_variant='critical', workflow='synthesis', model=object())
        self.assertEqual(rows[0]['status'], 'ok')
        self.assertEqual(rows[0]['successful_samples'], 2)
        self.assertEqual(list(rows[0]['annotation_stages']), ['initial', 'independent', 'sample_2', 'synthesis'])
        for _, prompts in calls[:3]:
            self.assertNotIn('independent_annotations', prompts[0])

    def test_insufficient_samples_are_failed_not_clean(self):
        def fake(items, prompts, **options):
            return [dict(id=identity(it), item=it, status='failed', result=None) for it in items]
        with patch('inference.run_oss_candidate_error_annotation.run_requests', side_effect=fake):
            row = annotate([item()], 'unused', prompt_variant='critical', workflow='synthesis', model=object())[0]
        self.assertEqual(row['status'], 'failed')
        self.assertIsNone(row['result'])
        self.assertEqual(row['successful_samples'], 0)

    def test_offline_reuses_one_model_and_keeps_stages_in_memory(self):
        model = object()
        def fake_generate(prompts, validators, **options):
            self.assertIs(options['model'], model)
            return [dict(parsed=validator({'spans': []}), attempts=[]) for validator in validators]
        with patch('inference.run_oss_SQM.init_oss_model', return_value=model) as initialize, \
             patch('inference.hard_generation.generate', side_effect=fake_generate) as generate:
            result = annotate([item()], 'unused', prompt_variant='critical', workflow='synthesis')
        self.assertEqual(result[0]['status'], 'ok')
        self.assertEqual(initialize.call_count, 1)
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(len(result[0]['annotation_stages']), 4)

    def test_empty_missing_and_online_requests_do_not_load_local_model(self):
        with patch('inference.run_oss_SQM.init_oss_model') as initialize:
            self.assertEqual(annotate([], 'unused'), [])
            missing = annotate([dict(item(), candidate=None)], 'unused')
            self.assertEqual(missing[0]['status'], 'failed')
            with patch('inference.candidate_error_generation.online_one',
                       return_value=dict(parsed={'spans': []}, attempts=[])) as online:
                rows = annotate([item()], 'unused', base_url='http://test',
                                prompt_variant='critical', workflow='synthesis')
            self.assertEqual(rows[0]['status'], 'ok')
            self.assertEqual(online.call_count, 4)
            initialize.assert_not_called()

    def test_bad_metadata_fails_before_loading_annotation_model(self):
        import numpy as np
        with patch('inference.run_oss_SQM.init_oss_model') as initialize:
            with self.assertRaises(TypeError):
                annotate([dict(item(), candidate_indices=[np.int64(0)])], 'unused')
            initialize.assert_not_called()

    def test_disallowed_workflow_fails_before_requests(self):
        for options in (dict(prompt_variant='wrong'), dict(workflow='synthesis'),
                        dict(workflow='synthesis', prompt_variant='critical', review=False), dict(samples=1)):
            with self.assertRaises(ValueError):
                annotate([item()], 'unused', **options)

    def test_data_entry_forwards_new_offline_options(self):
        from data.run_oss_candidate_error_annotation import main
        with patch('data.run_oss_candidate_error_annotation.load_items', return_value=[item()]), \
             patch('data.run_oss_candidate_error_annotation.annotate') as call:
            main('input.parquet', 'out', 'unused', prompt_variant='critical', workflow='synthesis',
                 samples=4, min_successful_samples=3, max_tokens=8192)
        options = call.call_args.kwargs
        self.assertIsNone(options['base_url'])
        self.assertEqual(options['samples'], 4)
        self.assertEqual(options['min_successful_samples'], 3)
        self.assertEqual(options['max_tokens'], 8192)


if __name__ == '__main__':
    unittest.main()
