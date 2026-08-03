import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.prepare_SFT_fused_flash_gpe_training_data import main as prepare


class _Tokenizer:
    def __init__(self, length=32):
        self.length = length

    def apply_chat_template(self, *args, **kwargs):
        return [0] * self.length


class FusedFlashGpeSftDataTest(unittest.TestCase):
    def setUp(self):
        self.candidates = ["你好", "您好", "嗨", "你好呀"]
        self.candidate_response = json.dumps(
            {"translations": self.candidates}, ensure_ascii=False
        )

    def _modern_frame(self):
        return pd.DataFrame({
            "src_text": ["Hello"],
            "src_lang": ["en"],
            "trg_lang": ["zh"],
            "flash_gpe_prompt_type": ["fixed_4"],
            "flash_gpe_max_candidates": [4],
            "flash_gpe_stage1_prompt": ["verbose candidate prompt"],
            "flash_gpe_stage1_thinking": ["candidate thinking"],
            "flash_gpe_stage1_response": [self.candidate_response],
            "flash_gpe_candidates": [self.candidates],
            "flash_gpe_stage2_prompt": ["verbose post-edit prompt"],
            "flash_gpe_stage2_thinking": ["post-edit thinking"],
            "flash_gpe_stage2_response": ["```translation\n您好\n```"],
            "flash_gpe_translation": ["您好"],
            "flash_gpe_parser_valid": [True],
        })

    def _legacy_frame(self):
        return pd.DataFrame({
            "src_text": ["Hello"],
            "src_lang": ["en"],
            "trg_lang": ["zh"],
            "gpe_candidate_generation": ["single_call"],
            "gpe_prompt_type": ["fixed_4"],
            "gpe_sampling_n": [4],
            "gpe_stage1_prompts": [["legacy candidate prompt"]],
            "gpe_stage1_thinking": [["candidate thinking"]],
            "gpe_stage1_responses": [[self.candidate_response]],
            "gpe_stage1_translations": [self.candidates],
            "gpe_stage2_prompt": ["legacy post-edit prompt"],
            "gpe_stage2_thinking": ["post-edit thinking"],
            "gpe_stage2_response": ["```translation\n您好\n```"],
            "gpe_translation": ["您好"],
            "gpe_parser_valid": [True],
        })

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_builds_one_ordered_fused_task_from_current_schema(
        self, from_pretrained
    ):
        from_pretrained.return_value = _Tokenizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.parquet"
            destination = Path(temp_dir) / "train.parquet"
            self._modern_frame().to_parquet(source, index=False)
            prepare(str(source), str(destination), tokenizer_path="unused")
            result = pd.read_parquet(destination)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["task"], "fused_flash_gpe")
        messages = result.iloc[0]["messages"]
        prompt = messages[0]["content"]
        assistant = messages[1]["content"]
        self.assertNotIn("JSON", prompt)
        self.assertNotIn("<thinking>", prompt)
        self.assertNotIn("<response>", prompt)
        self.assertEqual(assistant.count("<thinking>"), 2)
        self.assertEqual(assistant.count("<response>"), 2)
        positions = [
            assistant.index("candidate thinking"),
            assistant.index(self.candidate_response),
            assistant.index("post-edit thinking"),
            assistant.index("```translation"),
        ]
        self.assertEqual(positions, sorted(positions))

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_accepts_legacy_single_call_rows(self, from_pretrained):
        from_pretrained.return_value = _Tokenizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "legacy.parquet"
            destination = Path(temp_dir) / "train.parquet"
            self._legacy_frame().to_parquet(source, index=False)
            prepare(str(source), str(destination), tokenizer_path="unused")
            result = pd.read_parquet(destination)
        self.assertEqual(list(result["task"]), ["fused_flash_gpe"])

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_filters_rows_over_max_length(self, from_pretrained):
        from_pretrained.return_value = _Tokenizer(length=32)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.parquet"
            destination = Path(temp_dir) / "train.parquet"
            self._modern_frame().to_parquet(source, index=False)
            prepare(
                str(source),
                str(destination),
                tokenizer_path="unused",
                max_length=16,
            )
            result = pd.read_parquet(destination)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
