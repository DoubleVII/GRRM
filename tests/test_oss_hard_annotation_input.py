"""Regression coverage for merging file inputs with explicit model configuration."""
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from data.run_oss_hard_annotation import load_items, main


class AnnotationInputTests(unittest.TestCase):
    def test_file_input_default_and_explicit_data_id(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root/'custom.parquet'
            pd.DataFrame([dict(src_text='A pun.', trg_text='双关语。', src_lang='en', trg_lang='zh')]).to_parquet(source)

            def fake_annotate(items, model_path, **options):
                self.assertEqual(model_path, '/explicit/oss')
                self.assertEqual(options['temperature'], 0.9)
                return [dict(x, status='ok', result=dict(analysis='routine', score=0, spans=[])) for x in items]

            for data_id, expected in ((None, 'custom'), ('renamed', 'renamed')):
                output = root/expected
                with patch('data.run_oss_hard_annotation.annotate', side_effect=fake_annotate):
                    main(data_id=data_id, output_dir=output, model_path='/explicit/oss', input_file=source)
                artifact = json.loads((output/f'{expected}.oss.difficulty_annotations.json').read_text())
                self.assertEqual(artifact['items'][0]['data_id'], expected)
                self.assertEqual(artifact['config']['input_file'], str(source))
                with patch('data.run_oss_hard_annotation.annotate') as gen:
                    with self.assertRaises(FileExistsError):
                        main(data_id=data_id, output_dir=output, model_path='/explicit/oss', input_file=source)
                    gen.assert_not_called()

    def test_invalid_configuration_fails_before_model_call(self):
        for options in (dict(input_file='x.parquet', data_dir='data'),
                        dict(input_file='x.parquet', data_id='a,b'), dict()):
            with self.assertRaises(ValueError):
                load_items(**options)
        with patch('data.run_oss_hard_annotation.annotate') as gen:
            with self.assertRaisesRegex(ValueError, 'model_path'):
                main(input_file='x.parquet', output_dir='unused')
            with self.assertRaisesRegex(ValueError, 'output_dir'):
                main(input_file='x.parquet', model_path='/explicit/oss')
            gen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
