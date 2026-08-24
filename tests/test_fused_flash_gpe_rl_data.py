import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.prepare_RL_fused_flash_gpe_training_data import main as prepare


class FusedFlashGpeRlDataTest(unittest.TestCase):
    def _write_input(self, path: Path):
        pd.DataFrame({
            "src_text": ["Hello", "Goodbye"],
            "trg_text": ["你好", "再见"],
            "src_lang": ["en", "en"],
            "trg_lang": ["zh", "zh"],
        }).to_parquet(path, index=False)

    def test_builds_prompt_only_rl_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.parquet"
            destination = Path(temp_dir) / "train.parquet"
            self._write_input(source)
            prepare(str(source), str(destination), max_samples=1, seed=1)
            result = pd.read_parquet(destination)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["data_source"], "TowerBlocks-MT-Fused-FlashGPE")
        self.assertEqual(row["ability"], "fused_flash_gpe")
        self.assertNotIn("messages", result.columns)
        prompt = row["prompt"][0]
        self.assertEqual(prompt["role"], "user")
        self.assertRegex(prompt["content"], r"exactly [2-8]")
        self.assertNotIn("JSON", prompt["content"])
        self.assertNotIn("<thinking>", prompt["content"])
        self.assertNotIn("<response>", prompt["content"])
        self.assertEqual(row["reward_model"]["ground_truth"], "")
        self.assertEqual(row["extra_info"]["ref_text"], "你好")
        self.assertEqual(row["extra_info"]["prompt_type"], "markdown")
        self.assertEqual(row["extra_info"]["max_candidates"], 8)

    def test_adaptive_prompt_and_optional_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.parquet"
            destination = Path(temp_dir) / "train.parquet"
            pd.DataFrame({
                "src_text": ["Hello"],
                "src_lang": ["en"],
                "trg_lang": ["zh"],
            }).to_parquet(source, index=False)
            prepare(
                str(source),
                str(destination),
                prompt_type="adaptive",
                max_candidates=8,
                include_reference_info=False,
            )
            row = pd.read_parquet(destination).iloc[0]

        self.assertIn("up to 8", row["prompt"][0]["content"])
        self.assertNotIn("ref_text", row["extra_info"])

    def test_rejects_incompatible_fixed_prompt_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.parquet"
            destination = Path(temp_dir) / "train.parquet"
            self._write_input(source)
            with self.assertRaisesRegex(ValueError, "max_candidates=4"):
                prepare(
                    str(source),
                    str(destination),
                    prompt_type="fixed_4",
                    max_candidates=8,
                )


if __name__ == "__main__":
    unittest.main()
