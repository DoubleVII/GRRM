"""Grounding, response recovery, and unconstrained offline/online prediction contract."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.candidate_error_generation import online_one
from inference.candidate_error_prompts import build_shared_prompt, BASE_VARIANTS as VARIANTS
from inference.run_inst_candidate_error_prediction import predict
from utils.candidate_error_response import parse_response, validate_compact


def item():
    return dict(data_id='tiny', row_id=1, candidate_model='provided', candidate_index=0,
                src_text='The price is ５ dollars. 价格为五元𠮷。', candidate='价格为十元。',
                src_lang='English', trg_lang='Chinese', ref_text='SECRET_REF', comment='SECRET_COMMENT')


def output(quote='５ dollars', score=9):
    return {'spans': [dict(source_span=quote, score=score, reason='The quantity is wrong.')]}


class QwenCandidateErrorTests(unittest.TestCase):
    def test_relaxed_envelope_and_keys(self):
        raw = '## ANALYSIS\nQuantity differs.\n## final-json\n```JSON\n' + json.dumps(
            {'SPANS': [{'Source-Span': '５ dollars', 'SCORE': '9', 'Reason': 'Wrong quantity.'}]}) + '\n```\nDone.'
        value = validate_compact(parse_response(raw), item())
        self.assertEqual(value['spans'][0]['confidence'], .9)
        self.assertEqual(value['spans'][0]['source_span'], '５ dollars')

    def test_unfenced_analysis_and_explicit_empty(self):
        self.assertEqual(parse_response('Analysis: {irrelevant}\nFinal: {"spans":[]}'), {'spans': []})
        self.assertEqual(validate_compact({'spans': []}, item()), {'spans': [], 'quote_normalizations': []})
        self.assertIsNone(parse_response('No errors found.'))
        self.assertIsNone(validate_compact(None, item()))

    def test_do_not_fall_back_to_analysis_on_malformed_final(self):
        raw = 'Example:\n```json\n{"spans": []}\n```\nFinal:\n```json\n{"spans": [}\n```'
        self.assertIsNone(parse_response(raw))
        self.assertIsNone(parse_response('```json\n{"spans": []}\n```\nFinal:\n```json\n{"spans": ['))

    def test_unicode_case_and_spacing_preserve_original_offsets(self):
        value = validate_compact(output('5 DOLLARS'), item())
        s = value['spans'][0]
        self.assertEqual(item()['src_text'][s['source_start']:s['source_end']], '５ dollars')
        self.assertEqual(len(value['quote_normalizations']), 1)
        chinese = validate_compact(output('价 格 为 五 元𠮷'), item())['spans'][0]
        self.assertEqual(chinese['source_span'], '价格为五元𠮷')

    def test_ambiguous_or_missing_quotes_reject_whole_response(self):
        for quote in ('absent', '', 'the'):
            it = dict(item(), src_text='the cat and the dog')
            self.assertIsNone(validate_compact(output(quote), it))
        self.assertIsNone(validate_compact(output('S'), dict(item(), src_text='ß')))

    def test_invalid_scores_and_incomplete_spans(self):
        for score in (True, False, -1, 11, 8.5, float('nan'), float('inf'), 'high'):
            self.assertIsNone(validate_compact(output(score=score), item()))
        self.assertIsNone(validate_compact({'spans': [output()['spans'][0], {}]}, item()))
        self.assertIsNone(validate_compact({'spans': [{'source_span': '５ dollars', 'score': 9}]}, item()))
        self.assertEqual(validate_compact(output(score=.75), item(), scale=1)['spans'][0]['confidence'], .75)
        self.assertIsNone(validate_compact({'spans': [], 'SPANS': []}, item()))

    def test_no_privileged_input_in_any_prompt(self):
        for variant in VARIANTS:
            prompt = build_shared_prompt(item(), variant)
            self.assertNotIn('SECRET_REF', prompt)
            self.assertNotIn('SECRET_COMMENT', prompt)
            self.assertIn(item()['candidate'], prompt)
        compact = build_shared_prompt(item())
        for field in ('occurrence', 'external_info_need', 'error_type'):
            self.assertNotIn(field, compact)

    def test_online_unconstrained_analysis_and_retry(self):
        raw = 'Thinking about meaning.\n```JSON\n' + json.dumps(output()) + '\n```'
        responses = [io.BytesIO(json.dumps({'choices': [{'message': {'content': text}}]}).encode())
                     for text in ('unparseable', raw)]
        with patch('inference.candidate_error_generation.urllib.request.urlopen', side_effect=responses) as call:
            result = online_one('prompt', lambda v: validate_compact(v, item()), base_url='http://test/v1',
                                model_path='test', backend='inst', retry=1, response_format='candidate_relaxed')
        self.assertEqual(result['parsed']['spans'][0]['confidence'], .9)
        for args in call.call_args_list:
            body = json.loads(args.args[0].data)
            self.assertNotIn('response_format', body)
            self.assertNotIn('guided_json', body)

    def test_offline_uses_same_prompt_parser_and_normalizer(self):
        from utils.difficulty_response import parse_prediction_response
        def fake_generate(prompts, validators, **kwargs):
            self.assertEqual(kwargs['response_format'], 'candidate_relaxed')
            self.assertFalse(kwargs.get('json_mode', False))
            self.assertIn('# Analysis', prompts[0])
            value = parse_prediction_response('Analysis\n```JSON\n'+json.dumps(output())+'\n```', kwargs['response_format'])
            return [{'parsed': validators[0](value)}]
        with tempfile.TemporaryDirectory() as tmp:
            with patch('inference.hard_generation.generate', side_effect=fake_generate):
                records = predict([item()], 'test', model=object())
            self.assertEqual(records[0]['status'], 'ok')
            self.assertEqual(records[0]['result']['spans'][0]['confidence'], .9)
            with self.assertRaises(TypeError):
                predict([item()], 'test', json_mode=True)


if __name__ == '__main__':
    unittest.main()
