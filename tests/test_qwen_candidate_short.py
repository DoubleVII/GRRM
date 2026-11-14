"""Compressed prompts preserve offline routing, grounding and score semantics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.candidate_error_prompts import SHORT_VARIANTS as VARIANTS, build_short_prompt
from inference.run_inst_candidate_error_prediction import predict
from tests.test_qwen_candidate_error import item, output


class ShortPredictionTests(unittest.TestCase):
    def test_offline_variants_preserve_minor_scores_and_exclude_privileged_data(self):
        for variant in VARIANTS:
            def generate(prompts, validators, **kwargs):
                self.assertEqual(prompts, [build_short_prompt(item(), variant)])
                context = json.loads(prompts[0].rsplit('\nNow check this translation:\n', 1)[1])
                self.assertEqual(set(context), {'src_text','candidate','src_lang','trg_lang'})
                self.assertNotIn('SECRET_REF',prompts[0])
                self.assertNotIn('SECRET_COMMENT',prompts[0])
                self.assertEqual(kwargs['response_format'],'candidate_relaxed')
                value = output(score=2)
                value['spans'][0]['severity'] = 'minor'
                good = validators[0](value)
                value['spans'][0]['source_span'] = 'absent quote'
                self.assertIsNone(validators[0](value))
                return [{'parsed':good}]
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                with patch('inference.hard_generation.generate',side_effect=generate), patch(
                        'inference.candidate_error_generation.online_one',side_effect=AssertionError('HTTP in offline path')):
                    records = predict([item()], 'model',prompt_variant=variant,
                                      model_factory=lambda:object())
                span = records[0]['result']['spans'][0]
                self.assertEqual((span['severity'],span['confidence']),('minor',.2))
                self.assertEqual(item()['src_text'][span['source_start']:span['source_end']],span['source_span'])


if __name__=='__main__':
    unittest.main()
