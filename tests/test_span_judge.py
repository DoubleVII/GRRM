import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from eval.span_judge import matching_prompt, parse_matching_response, validate_pair_decision


def sample():
    return dict(src_text='The firefighter hugged her son.', src_lang='en', trg_lang='zh')


def span(quote, explanation):
    return dict(source_span=quote, explanation=explanation, score=7)


class SpanJudgeTests(unittest.TestCase):
    def test_inst_core_batches_pairs_and_disables_native_thinking(self):
        from eval.difficulty_eval_core import evaluate
        def item(quotes):
            return dict(sample(), data_id='tiny', row_id=0, status='ok',
                        result=dict(analysis='difficulty', score=7,
                                    spans=[span(q, 'concrete decision') for q in quotes]))
        left = item(['firefighter', 'her son'])
        right = item(['firefighter', 'her son'])
        responses = [dict(parsed=dict(analysis='decision', match=value)) for value in (True, False, True, True)]
        with patch('eval.difficulty_eval_core.generate', return_value=responses) as mocked:
            items, summary = evaluate([left], [right], model_path='small', span_backend='inst',
                                      span_prompt_variant='pairwise')
        self.assertEqual(len(mocked.call_args.args[0]), 4)
        self.assertEqual(mocked.call_args.kwargs['backend'], 'inst')
        self.assertFalse(mocked.call_args.kwargs['enable_thinking'])
        self.assertEqual(items[0]['result']['pairs'], [[0,0],[1,0],[1,1]])
        self.assertEqual(len(items[0]['generation']['pair_results']), 4)
        self.assertEqual(summary['overall']['spans']['f1'], 1)

    def test_inst_failed_pair(self):
        from eval.difficulty_eval_core import evaluate
        item = dict(sample(), data_id='tiny', row_id=0, status='ok',
                    result=dict(analysis='difficulty', score=7, spans=[span('her son', 'gender')]))
        with patch('eval.difficulty_eval_core.generate', return_value=[dict(parsed=None)]):
            items, summary = evaluate([item], [item], model_path='small', span_backend='inst')
        self.assertEqual(items[0]['span_status'], 'failed')
        self.assertEqual(summary['overall']['spans']['failed'], 1)

    def test_prompt_data_has_quotes_and_explanations_without_labels(self):
        source = dict(sample(), pairs=[[0, 0]], llm=1, proxy=1)
        prompt = matching_prompt(source, [span('her son', 'gender inference')],
                                 [span('firefighter', 'unexpected female referent')])
        payload = json.loads(prompt[prompt.rfind('\n')+1:])
        self.assertEqual(set(payload), {'src_text','src_lang','trg_lang','predicted_spans','reference_spans'})
        self.assertEqual(set(payload['predicted_spans'][0]), {'source_span','explanation'})
        self.assertIn('gender inference', prompt)
        self.assertNotIn('"llm"', prompt)
        self.assertNotIn('"score"', prompt)

    def test_pairwise_response_and_boundaries(self):
        for match in (True, False):
            result = dict(analysis='brief reason', match=match)
            self.assertEqual(parse_matching_response(json.dumps(result)), result)
        for value in (dict(analysis='same', match=1), dict(analysis='same', match='true'), dict(match=True)):
            self.assertIsNone(validate_pair_decision(value))
        self.assertIsNone(parse_matching_response('{"match":true}'))
        self.assertIsNone(parse_matching_response('not JSON'))
        with self.assertRaises(ValueError):
            matching_prompt(sample(), [], [])

    def test_comparison_uses_saved_matching_with_failure_coverage(self):
        from eval.benchmark_span_judges import evaluate_output
        rows = []
        outputs = []
        for i, (label, predicted) in enumerate(((1., 1.), (0., 0.), (0.5, None))):
            rows.append(dict(id=str(i), group=str(i), llm=label, proxy=label, model=0,
                             data_id='tiny', src_lang='en', prediction=['x'], reference=['x'],
                             pairs=[[0,0]] if label else []))
            outputs.append(dict(id=str(i), status='failed' if predicted is None else 'ok',
                                metrics=None if predicted is None else dict(f1=predicted),
                                result=None if predicted is None else dict(pairs=[[0,0]] if predicted else [])))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'test.pairwise.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in outputs))
            result = evaluate_output(rows, path)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['overall']['n'], 2)
        self.assertEqual(result['failure_as_zero']['n'], 3)
        self.assertEqual(result['overall']['mae'], 0)
        self.assertAlmostEqual(result['failure_as_zero']['mae'], 1/6)


if __name__ == '__main__':
    unittest.main()
