"""Merged execution keeps file ownership and submits all candidates once."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from data.merge_run_oss_candidate_error_task import main
from utils.candidate_error_protocol import identity, read_jsonl


class DirectoryTests(unittest.TestCase):
    def inputs(self, root, collision=False):
        for filename, dataset, candidates in [('a.parquet','a',['one','two']),
                                               ('nested/b.parquet','a' if collision else 'b',[None])]:
            path=root/filename
            path.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame([dict(data_id=dataset,row_id=0,candidate_index=i,candidate_model='inst',
                               src_text='source',candidate=c,src_lang='en',trg_lang='zh',
                               ref_text='reference',comment='secret reference evidence')
                          for i,c in enumerate(candidates)]).to_parquet(path,index=False)

    def test_single_call_then_split_annotation_and_prediction(self):
        for task in ('annotation','prediction'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.inputs(root/'input')
                def infer(items, model_path, **options):
                    self.assertEqual(len(items),3)
                    self.assertEqual(model_path,'test-model')
                    for item in items:
                        self.assertEqual('ref_text' in item,task=='annotation')
                        self.assertEqual('comment' in item,task=='annotation')
                    # Route by identity, never rely on inference order.
                    return [dict(id=identity(item),item=item,status='failed' if item['candidate'] is None else 'ok',
                                 result=None if item['candidate'] is None else {'spans':[]},
                                 annotation_stages={'initial':{'status':'ok'}}) for item in reversed(items)]
                with patch('data.merge_run_oss_candidate_error_task.annotate' if task=='annotation' else
                           'data.merge_run_oss_candidate_error_task.predict',side_effect=infer) as call:
                    main(root/'input',root/'out','test-model',task)
                    self.assertEqual(call.call_count,1)
                name='final.jsonl' if task=='annotation' else 'predictions.jsonl'
                a=read_jsonl(root/'out/a'/name);b=read_jsonl(root/'out/nested/b'/name)
                self.assertEqual([r['id'] for r in a],['a:0:inst:0','a:0:inst:1'])
                self.assertEqual(b[0]['status'],'failed')
                self.assertIn('annotation_stages',a[0])
                self.assertTrue((root/'out/merged_summary.json').exists())

    def test_collisions_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root/'input',collision=True)
            with patch('data.merge_run_oss_candidate_error_task.predict') as call:
                with self.assertRaisesRegex(ValueError,'Duplicate candidate identity'):
                    main(root/'input',root/'out','test-model','prediction')
                call.assert_not_called()

    def test_invalid_result_coverage_writes_no_dataset_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root/'input')
            with patch('data.merge_run_oss_candidate_error_task.predict',return_value=[]):
                with self.assertRaisesRegex(ValueError,'Result identities'):
                    main(root/'input',root/'out','test-model','prediction')
            self.assertFalse((root/'out').exists())

if __name__=='__main__':unittest.main()
