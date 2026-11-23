import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.prepare_RL_GQM_training_data import construct_data_item, main


class RLGQMEvidenceTest(unittest.TestCase):
    def test_shuffle_keeps_evidence_and_score_alignment(self):
        items = construct_data_item(
            "source", ["one", "two"], "en", "zh", "zh", "reference", "analysis", [7, 9],
            shuffle_augment=1, include_reference_prompt=True,
            comment=" expert evidence ", include_comment_prompt=True,
        )
        self.assertEqual(len(items), 2)
        for item in items:
            prompt = item["prompt"][0]["content"]
            self.assertIn("Chinese reference:\n```\nreference", prompt)
            self.assertIn("Expert comment:\n```\nexpert evidence", prompt)
            order = item["extra_info"]["shuffle_indices"]
            scores = json.loads(item["reward_model"]["ground_truth"])
            self.assertEqual(list(scores.values()), [[7, 9][i] for i in order])

    def test_disabled_missing_and_duplicate_evidence(self):
        args = ("source", ["one", "two"], "en", "zh", "zh", "one", "analysis", [7, 9])
        item = construct_data_item(*args, include_reference_prompt=True,
                                   comment=pd.NA, include_comment_prompt=True)[0]
        self.assertFalse(item["extra_info"]["reference_prompt_used"])
        self.assertFalse(item["extra_info"]["comment_prompt_used"])
        prompt = construct_data_item(*args, comment="evidence")[0]["prompt"][0]["content"]
        self.assertNotIn("Expert comment:", prompt)
        self.assertNotIn("Chinese reference:", prompt)

    def test_parquet_custom_columns_subgroups_and_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = str(Path(tmp) / "input.parquet")
            target = str(Path(tmp) / "output.parquet")
            pd.DataFrame({
                "src_text": ["source", "source2"], "src_lang": ["en"] * 2, "trg_lang": ["zh"] * 2,
                "mts": [["one", "two", "three"]] * 2, "scores": [[7, 9, 8]] * 2,
                "analysis": ["analysis"] * 2, "reference": ["ref", None], "expert": ["evidence", None],
            }).to_parquet(source)
            main(source, target, "mts", "scores", "analysis", subgroup_augment=3,
                 shuffle_augment=1, include_reference_prompt=True, include_comment_prompt=True,
                 ref_key="reference", comment_key="expert")
            rows = pd.read_parquet(target).to_dict("records")
            self.assertEqual(len(rows), 16)
            for row in rows:
                info = row["extra_info"]
                prompt = row["prompt"][0]["content"]
                self.assertEqual("Expert comment:" in prompt, bool(info["comment_prompt_used"]))
                self.assertEqual("Chinese reference:" in prompt, bool(info["reference_prompt_used"]))
            with self.assertRaisesRegex(ValueError, "missing"):
                main(source, target, "mts", "scores", "analysis",
                     include_comment_prompt=True, comment_key="missing")


if __name__ == "__main__":
    unittest.main()
