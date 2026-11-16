"""Regression tests for exact locations, leakage, matching, and threshold semantics."""
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.candidate_error_prompts import build_prompt
from inference.candidate_error_generation import run_requests,online_one
from utils.candidate_error_protocol import validate, locate
from eval.candidate_error_core import (aggregate, average_precision, best_threshold,
    bootstrap, category_mapping, confidence_labels, curve, location_edges, score,reliability)
from inference.run_oss_candidate_error_matching import matching_prompt, match
from data.run_oss_candidate_error_matching import prepare_rows
from eval.run_candidate_error_eval import evaluate
from data.candidate_error_input import load_items
from utils.candidate_error_protocol import read_predictions, identity, write_jsonl
from eval.candidate_error_contrast import evaluate_contrast
from scripts.complete_candidate_error_pilot import shared_graphs
from inference.run_inst_candidate_error_prediction import predict as predict_errors
from inference.run_oss_candidate_error_annotation import annotate as annotate_errors
from scripts.report_candidate_error_pilot import major_gate


def span(quote='词', start=0, severity='major', confidence=.8):
    return dict(source_span=quote,source_start=start,source_end=start+len(quote),
        severity=severity,confidence=confidence,error_type='meaning',external_info_need='unclear',explanation='wrong meaning')


def item():
    return dict(data_id='test',row_id=0,candidate_model='inst',candidate_index=0,
        src_text='词𠮷词',candidate='wrong',src_lang='zh',trg_lang='en',ref_text='SECRET_REFERENCE',
        comment='SECRET_COMMENT',annotation='SECRET_ANNOTATION',cluster='c',split='calibration',cohort='natural',family='wmt')


def row(p=None,g=None,edges=None):
    return dict(id='x',item=item(),predicted=p if p is not None else [span()],
                gold=g if g is not None else [span()],edges=edges if edges is not None else [[0,0]],prediction_status='ok')


class CandidateErrorTests(unittest.TestCase):
    def test_online_analysis_codeblock_parsing_and_retry_envelope(self):
        answer='# Step-by-step Analysis\nA concrete analysis.\n# Final JSON\n```json\n{"spans": []}\n```'
        responses=[io.BytesIO(json.dumps({'choices':[{'message':{'content':'bad envelope'}}]}).encode()),
                   io.BytesIO(json.dumps({'choices':[{'message':{'content':answer}}]}).encode())]
        with patch('inference.candidate_error_generation.urllib.request.urlopen',side_effect=responses) as request:
            result=online_one('prompt',lambda v:v,base_url='http://test/v1',model_path='test',backend='inst',
                              response_format='analysis_codeblock',format_feedback=True,retry=1)
            self.assertEqual(result['parsed'],{'spans':[]})
            body=json.loads(request.call_args_list[1].args[0].data)
            self.assertNotIn('response_format',body)
            self.assertIn('# Final JSON',body['messages'][0]['content'])

    def test_unicode_exact_offsets(self):
        x=locate({'source_span':'𠮷'},'source','词𠮷词')
        self.assertEqual((x['source_start'],x['source_end']),(1,2))

    def test_repeated_quote_requires_occurrence(self):
        with self.assertRaises(ValueError): locate({'source_span':'词'},'source','词𠮷词')
        self.assertEqual(locate({'source_span':'词','source_occurrence':1},'source','词𠮷词')['source_start'],2)

    def test_wrong_quote_and_offset_rejected(self):
        for x in ({'source_span':'語'},{'source_span':'词','source_start':1},{'source_span':'词','source_start':0,'source_end':2}):
            with self.assertRaises(ValueError): locate(x,'source','词𠮷词')

    def test_validate_probability_and_null(self):
        s=span()
        self.assertIsNotNone(validate({'spans':[s]},item(),scoring='probability'))
        for value in (True,float('nan'),-1,1.01,'high'):
            self.assertIsNone(validate({'spans':[dict(s,confidence=value)]},item(),scoring='probability'))
        self.assertIsNone(validate({'spans':[dict(s,source_span=None,source_start=None,source_end=None)]},item()))

    def test_addition_and_omission(self):
        addition=dict(span(),source_span=None,source_start=None,source_end=None,error_type='addition')
        self.assertIsNotNone(validate({'spans':[addition]},item()))
        omission=dict(span(),error_type='omission',candidate_span=None)
        self.assertIsNotNone(validate({'spans':[omission]},item()))

    def test_empty_is_valid_failed_is_not_empty(self):
        self.assertEqual(validate({'spans':[]},item()),{'spans':[]})
        self.assertIsNone(validate(None,item()))

    def test_prediction_allowlist(self):
        prompt=build_prompt(item())
        for forbidden in ('SECRET_REFERENCE','SECRET_COMMENT','SECRET_ANNOTATION'):
            self.assertNotIn(forbidden,prompt)
        self.assertIn('SECRET_REFERENCE',build_prompt(item(),annotation=True))

    def test_matching_hides_confidence_severity_repair(self):
        prompt=matching_prompt(item(),[dict(span(),explanation='SECRET_REPAIR')],[span()])
        context=json.loads(prompt.split('\n',1)[1])
        self.assertNotIn('severity',context['predicted'][0])
        self.assertNotIn('confidence',context['predicted'][0])
        self.assertNotIn('SECRET_REPAIR',prompt)

    def test_duplicates_one_tp(self):
        r=row([span(),span()],[span()],[[0,0],[1,0]])
        m=aggregate([r],.5,'probability')
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,1,0))
        self.assertEqual([x['label'] for x in confidence_labels([r],'probability')],[1,0])

    def test_high_first_matching_can_augment(self):
        r=row([span(confidence=.9),span(confidence=.8)],[span(),span('𠮷',1)],[[0,0],[0,1],[1,0]])
        self.assertEqual([x['label'] for x in confidence_labels([r],'probability')],[1,1])

    def test_minor_as_major_is_false_positive(self):
        r=row([span()],[span(severity='minor')])
        m=aggregate([r],.5,'probability')
        self.assertEqual((m['tp'],m['fp'],m['fn']),(0,1,0))

    def test_major_probability_can_override_severity(self):
        r=row([span(severity='minor')])
        self.assertEqual(aggregate([r],.5,'probability')['tp'],1)
        self.assertEqual(aggregate([r],.5,'severity')['fn'],1)

    def test_secondary_major_gate_remaps_edges_and_preserves_misses(self):
        r=row([span(severity='minor'),span('𠮷',1)],[span(),span('𠮷',1)],[[0,0],[1,1]])
        result=aggregate(major_gate([r]),.5,'probability')
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,0,1))

    def test_empty_gold_not_true_positive(self):
        m=aggregate([row([],[],[])],.5,'probability')
        self.assertEqual(m['tp'],0)
        self.assertIsNone(m['f1'])
        self.assertEqual(m['no_error_false_alarm_rate'],0)

    def test_failed_predictions_count_false_negatives(self):
        r=dict(row([],edges=[]),prediction_status='failed')
        m=aggregate([r],.5,'probability')
        self.assertEqual(m['fn'],1)
        self.assertEqual(m['prediction_failures'],1)

    def test_ap_includes_unproposed_gold(self):
        r=row([span()],[span(),span('𠮷',1)],[[0,0]])
        self.assertAlmostEqual(average_precision(curve([r],'probability')),.5)

    def test_equal_f1_prefers_fewer_calls(self):
        r=row([span(confidence=.9),span(confidence=.5)],[span(),span('𠮷',1),span('词',2)],[[0,0],[1,1]])
        self.assertEqual(best_threshold([r],'probability'),.5)

    def test_budget_and_char_union(self):
        r=row([span('词𠮷',0),span('𠮷词',1)],[span()],[[0,0]])
        m=aggregate([r],0,'probability')
        self.assertEqual(m['character_coverage'],1)
        self.assertEqual(m['triggers_per_100'],200)

    def test_calibrated_categories_monotone(self):
        rows=[row([span(confidence='low')]),row([span(confidence='high')],[],[])]
        m=category_mapping(rows)
        self.assertLessEqual(m['low'],m['high'])

    def test_bootstrap_reproducible(self):
        self.assertEqual(bootstrap([row()],.5,'probability',n=20),bootstrap([row()],.5,'probability',n=20))

    def test_confidence_correlation_and_constant_score_diagnostic(self):
        rows=[row([span(confidence=.9)]),row([span(confidence=.1)],[],[])]
        self.assertGreater(reliability(rows,'probability')['span_correctness_spearman'],.99)
        constant=reliability([row()],'probability')
        self.assertIsNone(constant['span_correctness_spearman'])
        self.assertEqual(constant['correlation_reason'],'constant_score_or_correctness')

    def test_repeated_calls_generate_again_without_resume(self):
        result = {'parsed': {'spans': [span()]}, 'attempts': []}
        with patch('inference.candidate_error_generation.online_one', return_value=result) as generate:
            first = run_requests([item()], ['prompt'], model_path='test', base_url='http://test')
            second = run_requests([item()], ['prompt'], model_path='test', base_url='http://test')
        self.assertEqual(first, second)
        self.assertEqual(generate.call_count, 2)

    def test_flat_parquet_candidates_take_precedence_over_legacy_samples(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'flat.parquet'
            pd.DataFrame([
                dict(src_text='词',candidate='flat',mt_sampling_text=['old','samples'],
                     src_lang='zh',trg_lang='en'),
                dict(src_text='字',candidate=None,mt_sampling_text=['legacy fallback'],
                     src_lang='zh',trg_lang='en'),
            ]).to_parquet(path)
            items=load_items(path)
            self.assertEqual([x['candidate'] for x in items],['flat',None])
            self.assertEqual([x['row_id'] for x in items],[0,1])
            self.assertEqual([x['candidate_index'] for x in items],[0,0])
            self.assertEqual(load_items(path,max_samples=1),items[:1])

    def test_parquet_loader_and_source_limit(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.parquet'
            pd.DataFrame([dict(src_text='词',trg_text='word',src_lang='zh',trg_lang='en',mt_sampling_text=['word','term']),
                          dict(src_text='字',trg_text='character',src_lang='zh',trg_lang='en',mt_sampling_text=['letter'])]).to_parquet(path)
            items=load_items(path,require_reference=True,max_samples=1)
            self.assertEqual(len(items),2)
            self.assertEqual([x['candidate_index'] for x in items],[0,1])
            self.assertEqual(items[0]['ref_text'],'word')
            prediction_items=load_items(path,max_samples=1)
            self.assertNotIn('ref_text',prediction_items[0])

    def test_upstream_failures_remain_in_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.jsonl'
            path.write_text('')
            Path(str(path)+'.proposal_failures.json').write_text(json.dumps([{'id':'failed','status':'failed','result':None}]))

    def test_null_candidate_is_failure_without_model_calls(self):
        it=dict(item(),candidate=None)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prediction.jsonl'
            predictions=predict_errors([it],'unused',base_url='http://invalid')
            self.assertEqual(predictions[0]['status'],'failed')
            silver=annotate_errors([it],'unused',base_url='http://invalid')
            self.assertEqual(silver[0]['status'],'failed')
            self.assertIsNone(silver[0]['result'])

    def test_end_to_end_calibration_ignores_test_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            predictions=[];gold=[]
            for i,split in enumerate(('calibration','test')):
                it=dict(item(),row_id=i,src_text='词𠮷词'+str(i),split=split,cluster=str(i))
                predictions.append(dict(id=identity(it),item=it,status='ok',result={'spans':[span(confidence=.7 if i==0 else .1)]}))
                gold.append(dict(id=identity(it),item=it,status='ok',result={'spans':[span()]}))
            def write(path,rows):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write(root/'p.jsonl',predictions);write(root/'g.jsonl',gold)
            write_jsonl(root/'matching.jsonl', match(predictions, gold, match_mode='exact'))
            split_file = root/'split.json'
            split_file.write_text(json.dumps({'calibration': [predictions[0]['id']], 'test': [predictions[1]['id']]}))
            evaluate(root/'matching.jsonl', root/'first', split_file=split_file, n_bootstrap=20)
            first=json.loads((root/'first/calibration.json').read_text())
            predictions[1]['result']['spans'][0]['confidence']=.99
            write(root/'p.jsonl',predictions)
            write_jsonl(root/'matching.jsonl', match(predictions, gold, match_mode='exact'))
            evaluate(root/'matching.jsonl', root/'second', split_file=split_file, n_bootstrap=20)
            second=json.loads((root/'second/calibration.json').read_text())
            self.assertEqual(first['threshold'],second['threshold'])
            self.assertEqual(first['threshold'],.7)

    def test_shared_graph_preserves_duplicate_penalty(self):
        it=item()
        a=dict(id='a',item=it,status='ok',result={'spans':[span(),span()]})
        b=dict(id='a',item=it,status='ok',result={'spans':[span()]})
        g=dict(id='a',item=it,status='ok',result={'spans':[span()]})
        graph=dict(id='a',item=it,predicted=[span()],gold=[span()],edges=[[0,0]],prediction_status='ok')
        with tempfile.TemporaryDirectory() as tmp:
            with patch('scripts.complete_candidate_error_pilot.prepare_rows',return_value=([graph],{})):
                variants=shared_graphs({'duplicates':[a],'single':[b]},[g],Path(tmp))
            m=aggregate(variants['duplicates'],.5,'probability')
            self.assertEqual((m['tp'],m['fp']),(1,1))

    def test_contrast_requalification_and_candidate_free_control(self):
        anchor=dict(source_span='词',candidate_model='inst',old_pass_index=0,old_fail_index=1)
        correct=dict(item(),cohort='contrast',split='test',contrast_anchor=anchor)
        wrong=dict(correct,candidate_index=1,candidate='incorrect')
        gold=[dict(id='correct',item=correct,status='ok',result={'spans':[]}),
              dict(id='wrong',item=wrong,status='ok',result={'spans':[span()]})]
        dependent=[dict(gold[0]),dict(gold[1])]
        sourcefree=[dict(gold[0],result={'spans':[span()]}),dict(gold[1])]
        failed_correct=[dict(gold[0],status='failed',result=None),dict(gold[1])]
        def fake_requests(items,prompts,**kwargs):
            # All union spans identify the shared source anchor.
            self.assertEqual(len(items),1)
            return [dict(id='correct',status='ok',result={'analysis':'same location','pairs':[[0,0]]})]
        settings={name:dict(scoring='probability',threshold=.5,mapping=None) for name in ('dependent','sourcefree','failed_correct')}
        with tempfile.TemporaryDirectory() as tmp:
            with patch('eval.candidate_error_contrast.run_requests',side_effect=fake_requests):
                reports=evaluate_contrast(gold,dict(dependent=dependent,sourcefree=sourcefree,failed_correct=failed_correct),settings,tmp,'test')
            self.assertEqual(reports['dependent']['test']['joint'],1)
            self.assertEqual(reports['sourcefree']['test']['joint'],0)
            self.assertEqual(reports['sourcefree']['test']['tie'],1)
            self.assertEqual(reports['failed_correct']['test']['joint'],0)
            self.assertEqual(reports['failed_correct']['test']['ranked_correctly'],0)
            self.assertIsNone(reports['failed_correct']['test']['correct_false_alarm'])


if __name__=='__main__': unittest.main()
