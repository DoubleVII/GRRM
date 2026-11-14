"""Target expansion preserves costs and one-to-one location matching."""
import unittest
from scripts.report_candidate_all_severities import measure, all_output_rows, any_curve


def span(start, severity, confidence):
    return dict(source_span='abc'[start:start+1], source_start=start, source_end=start+1,
                severity=severity, confidence=confidence)


def row(predicted, golden, edges):
    return dict(id='tiny', item=dict(src_text='abc',cluster='tiny',split='test',cohort='natural'),
                prediction_status='ok', predicted=predicted, gold=golden, edges=edges)


class AllSeveritiesTests(unittest.TestCase):
    def test_minor_target_recovers_fp_but_increases_gold_denominator(self):
        r=row([span(0,'minor',.3)], [span(0,'minor',1),span(1,'major',1)], [[0,0]])
        m=measure([r],'probability',.1)
        self.assertEqual((m['major']['tp'],m['major']['fp'],m['major']['fn']),(0,1,1))
        self.assertEqual((m['all']['tp'],m['all']['fp'],m['all']['fn']),(1,0,1))
        self.assertEqual(m['net_extra_tp'],1)
        self.assertEqual(m['major']['triggers_per_100'],m['all']['triggers_per_100'])

    def test_shared_location_cannot_claim_two_gold_errors(self):
        r=row([span(0,'major',.9)], [span(0,'major',1),span(1,'minor',1)], [[0,0],[0,1]])
        m=measure([r],'probability',.1)
        self.assertEqual(m['major']['tp'],1)
        self.assertEqual(m['all']['tp'],1)
        self.assertEqual(m['all']['fn'],1)
        self.assertLess(m['all']['f1'],m['major']['f1'])

    def test_all_outputs_includes_minor_predictions_in_major_fp(self):
        r=row([span(0,'major',.9),span(1,'minor',.1)],
              [span(0,'major',1),span(1,'minor',1)], [[0,0],[1,1]])
        original=measure([r],'severity',1)
        all_outputs=measure(all_output_rows([r]),'probability',.5)
        all_calibrated=measure([r],'severity',0)
        self.assertEqual(original['all']['tp'],1)
        self.assertEqual(all_outputs['major']['fp'],1)
        self.assertEqual(all_outputs['all']['tp'],2)
        self.assertEqual(all_calibrated['all'],all_outputs['all'])
        points=any_curve([r],'severity')
        self.assertEqual(max(points,key=lambda x:x['f1'])['threshold'],0)
        self.assertEqual(r['predicted'][1]['confidence'],.1)


if __name__=='__main__':
    unittest.main()
