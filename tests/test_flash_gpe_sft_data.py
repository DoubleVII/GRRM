import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.flash_gpe_sft_data_utils import (
    normalize_flash_gpe_row,
    validate_flash_gpe_record,
)
from data.run_oss_flash_gpe_sft_data import main as collect_flash_gpe
from scripts.prepare_SFT_flash_gpe_training_data import main as prepare_flash_gpe


class FlashGpeSftDataTest(unittest.TestCase):
    def setUp(self):
        self.candidates = ["你好", "您好", "嗨", "你好呀"]
        self.candidate_response = json.dumps(
            {"translations": self.candidates}, ensure_ascii=False
        )

    def test_validator_requires_joint_candidates_and_post_edit(self):
        kwargs = {
            "candidate_response": self.candidate_response,
            "candidate_thinking": "Generate four candidates.",
            "candidates": self.candidates,
            "post_edit_response": "```translation\n您好\n```",
            "post_edit_thinking": "Choose the formal candidate.",
            "post_edit_translation": "您好",
            "max_candidates": 4,
            "prompt_type": "fixed_4",
        }
        self.assertTrue(validate_flash_gpe_record(**kwargs))
        self.assertFalse(
            validate_flash_gpe_record(
                **{**kwargs, "candidates": self.candidates[:3]}
            )
        )

    @patch("data.run_oss_flash_gpe_sft_data.run_pipeline")
    @patch("data.run_oss_flash_gpe_sft_data.init_oss_model")
    def test_collector_writes_flash_gpe_schema(self, init_model, pipeline):
        init_model.return_value = object()
        pipeline.return_value = {
            "candidate_generation": {
                "prompts": ["oss candidate prompt"],
                "translations": [self.candidates],
                "responses": [self.candidate_response],
                "thinking": ["Generate four candidates."],
            },
            "usable_candidate_counts": [4],
            "post_edit": {
                "translations": ["您好"],
                "responses": ["```translation\n您好\n```"],
                "thinking": ["Choose the formal candidate."],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.parquet"
            output_path = Path(temp_dir) / "output.parquet"
            pd.DataFrame({
                "src_text": ["Hello"],
                "src_lang": ["en"],
                "trg_lang": ["zh"],
            }).to_parquet(input_path, index=False)
            collect_flash_gpe(str(input_path), str(output_path))
            result = pd.read_parquet(output_path).iloc[0]
        self.assertEqual(result["sft_method"], "flash_gpe")
        self.assertEqual(result["flash_gpe_prompt_type"], "fixed_4")
        self.assertEqual(result["flash_gpe_max_candidates"], 4)
        self.assertEqual(result["flash_gpe_stage1_response"], self.candidate_response)
        self.assertEqual(list(result["flash_gpe_candidates"]), self.candidates)

    def test_legacy_single_call_row_is_normalized(self):
        row = pd.Series({
            "gpe_candidate_generation": "single_call",
            "gpe_prompt_type": "fixed_4",
            "gpe_sampling_n": 4,
            "gpe_stage1_prompts": ["old prompt"],
            "gpe_stage1_thinking": ["thinking"],
            "gpe_stage1_responses": [self.candidate_response],
            "gpe_stage1_translations": self.candidates,
            "gpe_stage2_prompt": "old post-edit prompt",
            "gpe_stage2_thinking": "post-edit thinking",
            "gpe_stage2_response": "```translation\n您好\n```",
            "gpe_translation": "您好",
            "gpe_parser_valid": True,
        })
        normalized = normalize_flash_gpe_row(row)
        self.assertTrue(normalized["legacy"])
        self.assertEqual(normalized["candidate_response"], self.candidate_response)
        with self.assertRaisesRegex(ValueError, "neither FlashGPE"):
            normalize_flash_gpe_row(
                pd.Series({"gpe_candidate_generation": "independent"})
            )

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_preparation_builds_two_short_prompt_tasks(self, from_pretrained):
        class Tokenizer:
            @staticmethod
            def apply_chat_template(*args, **kwargs):
                return [0] * 32

        from_pretrained.return_value = Tokenizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.parquet"
            output_path = Path(temp_dir) / "train.parquet"
            pd.DataFrame({
                "src_text": ["Hello"],
                "src_lang": ["en"],
                "trg_lang": ["zh"],
                "flash_gpe_prompt_type": ["fixed_4"],
                "flash_gpe_max_candidates": [4],
                "flash_gpe_candidate_count": [4],
                "flash_gpe_stage1_prompt": ["verbose JSON prompt"],
                "flash_gpe_stage1_thinking": ["Generate candidates."],
                "flash_gpe_stage1_response": [self.candidate_response],
                "flash_gpe_candidates": [self.candidates],
                "flash_gpe_stage2_prompt": ["verbose format prompt"],
                "flash_gpe_stage2_thinking": ["Choose formal."],
                "flash_gpe_stage2_response": ["```translation\n您好\n```"],
                "flash_gpe_translation": ["您好"],
                "flash_gpe_parser_valid": [True],
            }).to_parquet(input_path, index=False)
            prepare_flash_gpe(
                str(input_path), str(output_path), tokenizer_path="unused"
            )
            candidate_data = pd.read_parquet(
                Path(temp_dir) / "train.flash_gpe_candidates.parquet"
            )
            flash_gpe_data = pd.read_parquet(
                Path(temp_dir) / "train.flash_gpe.parquet"
            )
        self.assertEqual(candidate_data.iloc[0]["task"], "flash_gpe_candidates")
        self.assertEqual(flash_gpe_data.iloc[0]["task"], "flash_gpe")
        prompts = [
            candidate_data.iloc[0]["messages"][0]["content"],
            flash_gpe_data.iloc[0]["messages"][0]["content"],
        ]
        for prompt in prompts:
            self.assertNotIn("JSON", prompt)
            self.assertNotIn("<thinking>", prompt)
            self.assertNotIn("<response>", prompt)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_preparation_filters_both_tasks_by_length(self, from_pretrained):
        class Tokenizer:
            @staticmethod
            def apply_chat_template(*args, **kwargs):
                return [0] * 32

        from_pretrained.return_value = Tokenizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.parquet"
            output_path = Path(temp_dir) / "train.parquet"
            pd.DataFrame({
                "src_text": ["Hello"],
                "src_lang": ["en"],
                "trg_lang": ["zh"],
                "flash_gpe_prompt_type": ["fixed_4"],
                "flash_gpe_max_candidates": [4],
                "flash_gpe_stage1_prompt": ["verbose JSON prompt"],
                "flash_gpe_stage1_thinking": ["Generate candidates."],
                "flash_gpe_stage1_response": [self.candidate_response],
                "flash_gpe_candidates": [self.candidates],
                "flash_gpe_stage2_prompt": ["verbose format prompt"],
                "flash_gpe_stage2_thinking": ["Choose formal."],
                "flash_gpe_stage2_response": ["```translation\n您好\n```"],
                "flash_gpe_translation": ["您好"],
                "flash_gpe_parser_valid": [True],
            }).to_parquet(input_path, index=False)
            prepare_flash_gpe(
                str(input_path),
                str(output_path),
                tokenizer_path="unused",
                max_length=16,
            )
            candidates = pd.read_parquet(
                Path(temp_dir) / "train.flash_gpe_candidates.parquet"
            )
            post_edit = pd.read_parquet(
                Path(temp_dir) / "train.flash_gpe.parquet"
            )

        self.assertTrue(candidates.empty)
        self.assertTrue(post_edit.empty)


if __name__ == "__main__":
    unittest.main()
