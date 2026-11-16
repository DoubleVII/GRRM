"""Matching artifacts are reusable across offline partitions and thresholds."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.run_oss_candidate_error_matching import match, matching_prompt
from data.merge_run_oss_candidate_error_matching import main as save_matching
from eval.run_candidate_error_eval import evaluate, load_matching, assign_splits
from tests.test_candidate_error_pipeline import item, span
from utils.candidate_error_protocol import identity, write_jsonl, read_jsonl


def records(data_id='test', index=0, **metadata):
    it = dict(item(), data_id=data_id, row_id=index, src_text='词𠮷词'+str(index), **metadata)
    p = dict(id=identity(it), item=it, status='ok', result={'spans': [span()]})
    g = dict(p, result={'spans': [span()]})
    return p, g


class MatchingTests(unittest.TestCase):
    def test_merged_entrypoint_requires_directories_and_rejects_file_options(self):
        with self.assertRaises(TypeError):
            save_matching(output_dir='unused', predictions_file='p.jsonl', silver_file='g.jsonl')
        with self.assertRaises(TypeError):
            save_matching(predictions_dir='p', silver_dir='g', output_dir='unused',
                          predictions_file='p.jsonl', silver_file='g.jsonl')

    def test_prompt_ignores_prediction_explanation_and_scores(self):
        p, _ = records()
        a = matching_prompt(p['item'], [span()], [span()])
        b = matching_prompt(p['item'], [dict(span(), explanation='wrong repair', severity='minor', confidence=.1)], [span()])
        self.assertEqual(a, b)
        self.assertNotIn('SECRET_REFERENCE', a)
        self.assertNotIn('SECRET_COMMENT', a)

    def test_batch_and_original_indices_no_file_io(self):
        ps, gs = zip(*(records(index=i) for i in range(3)))
        addition = dict(span(), source_span=None, source_start=None, source_end=None)
        ps = [dict(p, result={'spans': [addition, span()]}) for p in ps]
        def generate(prompts, validators, **options):
            self.assertEqual(len(prompts), 3)
            self.assertTrue(all(len(json.loads(p.split('\n', 1)[1])['predicted']) == 1 for p in prompts))
            return [dict(parsed=v({'analysis': 'same location', 'pairs': [[0, 0]]})) for v in validators]
        with patch('inference.hard_generation.generate', side_effect=generate) as call, \
                patch('builtins.open', side_effect=AssertionError('inference file IO')):
            rows = match(ps, list(gs), model_path='unused', model=object())
        self.assertEqual(call.call_count, 1)
        self.assertEqual([r['edges'] for r in rows], [[[1, 0]]]*3)
        self.assertTrue(all('split' not in r for r in rows))
        self.assertEqual([r['id'] for r in rows], [p['id'] for p in ps])

    def test_upstream_failures_empty_and_matching_failure(self):
        ps, gs = zip(*(records(index=i) for i in range(4)))
        ps, gs = list(ps), list(gs)
        ps[0] = dict(ps[0], status='failed', result=None)
        gs[1] = dict(gs[1], status='failed', result=None)
        ps[2] = dict(ps[2], result={'spans': []})
        with patch('inference.hard_generation.generate', return_value=[dict(parsed=None)]):
            rows = match(ps, gs, model_path='unused', model=object())
        self.assertEqual(len(rows), 4)
        self.assertEqual([r['matching_status'] for r in rows], ['skipped']*3+['failed'])
        self.assertEqual(rows[0]['prediction_status'], 'failed')
        self.assertEqual(rows[1]['excluded_reason'], 'gold_failed')
        self.assertTrue(all(not r['edges'] for r in rows))

    def test_invalid_inputs_fail_before_generation(self):
        p, g = records()
        with patch('inference.hard_generation.generate') as generate:
            for ps, gs in (([p, p], [g]), ([p], []), ([dict(p, id='bad')], [g]),
                           ([p], [dict(g, item=dict(g['item'], candidate='different'))])):
                with self.assertRaises(ValueError):
                    match(ps, gs, model_path='unused')
            generate.assert_not_called()

    def test_silver_ltb_group_is_preserved_and_conflicts_rejected(self):
        p, g = records()
        g = dict(g, item=dict(g['item'], ltb_id='shared'))
        self.assertEqual(match([p], [g], match_mode='exact')[0]['item']['ltb_id'], 'shared')
        with self.assertRaisesRegex(ValueError, 'ltb_id mismatch'):
            match([dict(p, item=dict(p['item'], ltb_id='other'))], [g], match_mode='exact')

    def test_joint_matching_equals_separate_matching(self):
        ps, gs = zip(*(records(name, i) for i, name in enumerate(('a', 'a', 'b'))))
        def generate(prompts, validators, **options):
            return [dict(parsed=v({'analysis': 'same', 'pairs': [[0, 0]]})) for v in validators]
        with patch('inference.hard_generation.generate', side_effect=generate):
            joint = match(list(ps), list(gs), model_path='unused')
            separate = match(list(ps[:2]), list(gs[:2]), model_path='unused') + \
                match(list(ps[2:]), list(gs[2:]), model_path='unused')
        self.assertEqual(joint, separate)

    def test_directory_batch_split_filter_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('a', 'b', 'train'):
                ps, gs = zip(*(records(name, i) for i in (2, 0)))
                write_jsonl(root/'p'/name/'predictions.jsonl', ps)
                write_jsonl(root/'g'/name/'final.jsonl', gs)
            kwargs = dict(predictions_dir=root/'p', silver_dir=root/'g', data_ids='b,a', output_dir=root/'out')
            summary = save_matching(**kwargs, dry_run=True)
            self.assertEqual(summary['semantic_requests'], 4)
            self.assertFalse((root/'out').exists())
            def generate(prompts, validators, **options):
                self.assertEqual(len(prompts), 4)
                return [dict(parsed=v({'analysis': '', 'pairs': [[0, 0]]})) for v in validators]
            with patch('inference.hard_generation.generate', side_effect=generate) as call:
                save_matching(**kwargs, model_path='unused')
            self.assertEqual(call.call_count, 1)
            self.assertFalse((root/'out/train').exists())
            for name in ('a', 'b'):
                self.assertEqual([r['item']['row_id'] for r in read_jsonl(root/'out'/name/'matching.jsonl')], [2, 0])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                save_matching(**kwargs)
            with self.assertRaisesRegex(ValueError, 'Missing'):
                save_matching(**dict(kwargs, data_ids='missing'), dry_run=True)

    def test_none_data_ids_discovers_all_datasets_and_requires_silver(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('b', 'a'):
                p, g = records(name)
                write_jsonl(root/'p'/name/'predictions.jsonl', [p])
                write_jsonl(root/'g'/name/'final.jsonl', [g])
            kwargs = dict(predictions_dir=root/'p', silver_dir=root/'g', output_dir=root/'out',
                          match_mode='exact')
            save_matching(**kwargs, data_ids=None)
            manifest = json.loads((root/'out/manifest.json').read_text())
            self.assertEqual([entry['data_id'] for entry in manifest['files']], ['a', 'b'])
            self.assertEqual(manifest['total'], 2)
            self.assertTrue(all((root/'out'/name/'matching.jsonl').is_file() for name in ('a', 'b')))
            (root/'g/b/final.jsonl').unlink()
            with patch('inference.hard_generation.generate') as generate:
                with self.assertRaisesRegex(ValueError, 'Missing prediction or silver'):
                    save_matching(**dict(kwargs, output_dir=root/'missing'), dry_run=True)
                generate.assert_not_called()
            (root/'empty').mkdir()
            with self.assertRaisesRegex(ValueError, 'No dataset'):
                save_matching(**dict(kwargs, predictions_dir=root/'empty'), dry_run=True)

    def test_offline_metrics_reuse_and_gold_failure_audit(self):
        ps, gs = zip(*(records(index=i) for i in range(3)))
        ps, gs = list(ps), list(gs)
        ps[0] = dict(ps[0], status='failed', result=None)
        gs[1] = dict(gs[1], status='failed', result=None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root/'matching.jsonl', match(ps, gs, match_mode='exact'))
            with patch('inference.hard_generation.generate', side_effect=AssertionError('offline called model')):
                rows, first = evaluate(root/'matching.jsonl', root/'low', threshold=.5, n_bootstrap=0)
                _, second = evaluate(root/'matching.jsonl', root/'high', threshold=.9, n_bootstrap=0)
            self.assertEqual(first['test.natural']['overall']['tp'], 1)
            self.assertEqual(first['test.natural']['overall']['fn'], 1)
            self.assertEqual(second['test.natural']['overall']['tp'], 0)
            self.assertEqual(len(rows), 3)
            self.assertIsNone(rows[1]['metrics'])
            self.assertEqual(json.loads((root/'low/report.json').read_text())['audit']['gold_failed'], 1)
            self.assertEqual(len(read_jsonl(root/'low/evaluation_rows.jsonl')), 3)
            self.assertEqual(len(json.loads((root/'low/rows.json').read_text())), 2)

    def test_manual_calibration_frozen_threshold_and_test_independence(self):
        ps, gs = zip(*(records(data_id=name, index=i) for i, name in enumerate(('cal', 'test', 'train'))))
        ps = list(ps)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root/'matching.jsonl', match(ps, list(gs), match_mode='exact'))
            kwargs = dict(calibration_data_ids='cal', test_data_ids='test', n_bootstrap=0)
            rows, _ = evaluate(root/'matching.jsonl', root/'one', **kwargs)
            self.assertEqual(rows[2]['evaluation_excluded_reason'], 'not_selected')
            frozen = json.loads((root/'one/calibration.json').read_text())
            self.assertEqual(frozen['threshold'], .8)
            ps[1] = dict(ps[1], result={'spans': [span(confidence=.1)]})
            write_jsonl(root/'matching.jsonl', match(ps, list(gs), match_mode='exact'))
            evaluate(root/'matching.jsonl', root/'two', **kwargs)
            self.assertEqual(frozen, json.loads((root/'two/calibration.json').read_text()))
            evaluate(root/'matching.jsonl', root/'frozen', test_data_ids='test',
                     calibration_file=root/'one/calibration.json', n_bootstrap=0)
            root.joinpath('split.json').write_text(json.dumps({'calibration': [ps[0]['id']], 'test': [ps[1]['id']]}))
            evaluate(root/'matching.jsonl', root/'ids', split_file=root/'split.json', n_bootstrap=0)
            self.assertEqual(frozen, json.loads((root/'ids/calibration.json').read_text()))

    def test_split_overlap_and_frozen_overlap(self):
        p, g = records('cal')
        q, h = records('test')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root/'matching.jsonl', match([p, q], [g, h], match_mode='exact'))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                evaluate(root/'matching.jsonl', root/'out', calibration_data_ids='cal', test_data_ids='test')
            p, g = records('cal', 1, ltb_id='shared')
            q, h = records('test', 2, ltb_id='shared')
            write_jsonl(root/'matching.jsonl', match([p, q], [g, h], match_mode='exact'))
            with self.assertRaisesRegex(ValueError, 'ltb_ids'):
                evaluate(root/'matching.jsonl', root/'out', calibration_data_ids='cal', test_data_ids='test')
            evaluate(root/'matching.jsonl', root/'cal', calibration_data_ids='cal', n_bootstrap=0)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                evaluate(root/'matching.jsonl', root/'test', test_data_ids='test',
                         calibration_file=root/'cal/calibration.json', n_bootstrap=0)

    def test_bad_partition_threshold_and_edges(self):
        p, g = records()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = match([p], [g], match_mode='exact')
            write_jsonl(root/'matching.jsonl', rows)
            for kwargs in ({}, {'threshold': float('nan')}, {'threshold': 2},
                           {'threshold': .5, 'test_data_ids': 'missing'},
                           {'threshold': .5, 'test_data_ids': 'test', 'calibration_data_ids': 'test'}):
                with self.assertRaises(ValueError):
                    evaluate(root/'matching.jsonl', root/'out', **kwargs)
            root.joinpath('split.json').write_text(json.dumps({'test': [p['id']], 'calibration': [p['id']]}))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                assign_splits(rows, split_file=root/'split.json')
            rows[0]['edges'] = [[8, 0]]
            write_jsonl(root/'matching.jsonl', rows)
            with self.assertRaisesRegex(ValueError, 'edges'):
                load_matching(root/'matching.jsonl')

    def test_eval_has_no_inference_dependency(self):
        source = Path('eval/run_candidate_error_eval.py').read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual((node.module or '').split('.')[0], 'inference')
        self.assertNotIn('model_path', source)

    def test_cli_boolean_dry_run_and_offline_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, g = records()
            write_jsonl(root/'p/test/predictions.jsonl', [p])
            write_jsonl(root/'g/test/final.jsonl', [g])
            args = [sys.executable, '-m', 'data.merge_run_oss_candidate_error_matching',
                    '--predictions_dir', str(root/'p'), '--silver_dir', str(root/'g'),
                    '--output_dir', str(root/'matches'), '--data_ids', 'test']
            subprocess.run([*args, '--dry_run', 'true'], check=True, capture_output=True)
            self.assertFalse((root/'matches').exists())
            subprocess.run([*args, '--dry_run', 'false', '--match_mode', 'exact'],
                           check=True, capture_output=True)
            env = dict(os.environ, PYTHON=sys.executable, MATCHING_INPUT=str(root/'matches'),
                       OUTPUT_DIR=str(root/'metrics'), THRESHOLD='0.5', N_BOOTSTRAP='0')
            for key in ('CALIBRATION_FILE', 'CALIBRATION_DATA_IDS', 'TEST_DATA_IDS', 'SPLIT_FILE', 'DATA_IDS'):
                env.pop(key, None)
            subprocess.run(['bash', 'run_candidate_error_eval.sh'], env=env,
                           check=True, capture_output=True)
            report = json.loads((root/'metrics/report.json').read_text())
            self.assertEqual(report['reports']['test.natural']['overall']['tp'], 1)

    def test_offline_table_recomputes_selected_and_balanced_metrics(self):
        from scripts.report_candidate_error_eval import table_rows
        from scripts.report_hard_eval_balanced import GROUPS
        names = [name for group in GROUPS.values() for name in group]
        ps, gs = zip(*(records(name, i) for i, name in enumerate(names)))
        ps = list(ps)
        # WMT group has one miss, making pooled and group-equal metrics differ.
        ps[0] = dict(ps[0], result={'spans': []})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root/'matching.jsonl', match(ps, list(gs), match_mode='exact'))
            evaluate(root/'matching.jsonl', root/'metrics', threshold=.5, n_bootstrap=0)
            path = root/'metrics/report.json'
            table = table_rows(path, data_ids=','.join(names[:2]))
            self.assertEqual(table[-1][1], 2)
            self.assertAlmostEqual(table[-1][4], 2/3)
            balanced = table_rows(path, balanced=True)
            self.assertEqual(balanced[-1][1], 7)
            self.assertAlmostEqual(balanced[-1][4], (2/3+1+1)/3)
            self.assertIsNone(balanced[-1][6])  # Constant score/correctness gives undefined rho.
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                table_rows(path, data_ids='missing')
            saved = read_jsonl(root/'metrics/evaluation_rows.jsonl')
            saved[0]['edges'] = []
            saved[0]['item']['candidate'] = 'tampered'
            write_jsonl(root/'metrics/evaluation_rows.jsonl', saved)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                table_rows(path)


if __name__ == '__main__':
    unittest.main()
