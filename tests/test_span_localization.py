import json
import unittest
from unittest.mock import patch

from eval.difficulty_eval_core import matching_prompt, evaluate


class SpanLocalizationTests(unittest.TestCase):
    def item(self, explanation):
        return dict(data_id='tiny', row_id=0, src_text='kick the bucket', src_lang='en', trg_lang='zh',
                    status='ok', result=dict(analysis='difficulty', score=6,
                    spans=[dict(source_span='kick the bucket', explanation=explanation, score=6)]))

    def test_localization_retains_explanations_but_does_not_grade_solutions(self):
        left, right = self.item('a physical container'), self.item('death idiom')
        prompt = matching_prompt(left, right, match_policy='localization')
        data = json.loads(prompt.rsplit('\n', 1)[-1])
        self.assertEqual(data['predicted_spans'][0]['explanation'], 'a physical container')
        self.assertIn('must NOT veto a location match', prompt)
        self.assertNotIn('"score"', prompt)
        self.assertEqual(matching_prompt(left, right), matching_prompt(left, right, match_policy='semantic'))

    def test_policy_reaches_oss_only_and_keeps_matching_protocol(self):
        left, right = self.item('a physical container'), self.item('death idiom')
        with patch('eval.difficulty_eval_core.generate', return_value=[dict(parsed=dict(analysis='same location', pairs=[[0,0]]))]) as generate:
            items, summary = evaluate([left],[right],model_path='oss',span_match_policy='localization')
        self.assertEqual(generate.call_args.kwargs['backend'],'oss')
        self.assertIn('must NOT veto a location match', generate.call_args.args[0][0])
        self.assertEqual(items[0]['result']['matching'],[[0,0]])
        self.assertEqual(summary['overall']['spans']['f1'],1)
        for backend in ['inst','heuristic']:
            with self.assertRaises(ValueError):
                evaluate([left],[right],span_backend=backend,span_match_policy='localization')


if __name__ == '__main__':
    unittest.main()
