"""Parquet trace metadata must serialize before costly inference begins."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data.candidate_error_input import load_items
from inference.candidate_error_generation import run_requests
from utils.candidate_error_protocol import digest, read_jsonl, write_jsonl


class CandidateSerializationTests(unittest.TestCase):
    def test_parquet_indices_and_missing_ltb_id_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(data_id='demo', row_id=i, candidate_model='inst', candidate_index=0,
                         src_text='source', candidate='translation', src_lang='en', trg_lang='zh',
                         ref_text='reference', candidate_indices=[0, 3], ltb_id=ltb_id)
                    for i, ltb_id in enumerate((7, None))]
            pd.DataFrame(rows).to_parquet(root / 'input.parquet', index=False)
            items = load_items(root / 'input.parquet', require_reference=True)
            self.assertEqual(items[0]['candidate_indices'], [0, 3])
            self.assertTrue(all(type(i) is int for i in items[0]['candidate_indices']))
            self.assertNotIn('ltb_id', items[1])
            json.dumps(items, allow_nan=False)
            digest(items)
            write_jsonl(root / 'items.jsonl', items)
            self.assertEqual(read_jsonl(root / 'items.jsonl'), items)

    def test_bad_metadata_fails_before_model_initialization(self):
        item = dict(data_id='demo', row_id=0, candidate_model='inst', candidate_index=0,
                    src_text='source', candidate='translation', src_lang='en', trg_lang='zh',
                    candidate_indices=[np.int64(0)])
        with patch('inference.hard_generation.generate') as generate:
            with self.assertRaisesRegex(TypeError, 'int64'):
                run_requests([item], ['prompt'], model_path='test', backend='oss')
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
