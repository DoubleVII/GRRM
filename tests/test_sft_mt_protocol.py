import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.prepare_SFT_GPE_training_data import _output_paths, main as prepare_gpe
from inference.run_oss_diverse_mt import extract_final_translation
from inference.sft_mt_protocol import (
    build_scd_followup_prompt,
    build_scd_stage1_prompt,
    build_sft_direct_prompt,
    build_sft_fused_flash_gpe_prompt,
    build_sft_gpe_prompt,
    build_sft_flash_gpe_candidate_prompt,
    format_fused_sft_output,
    format_sft_output,
    parse_fused_sft_output,
    parse_fused_task_output,
    parse_sft_output,
    parse_task_output,
)


class SftMtProtocolTest(unittest.TestCase):
    def test_fused_output_round_trip(self):
        text = format_fused_sft_output(
            "generate candidates",
            "candidate response",
            "compare candidates",
            "final response",
        )
        envelope = parse_fused_sft_output(text)
        self.assertEqual(envelope.candidate_thinking, "generate candidates")
        self.assertEqual(envelope.post_edit_response, "final response")
        parsed = parse_fused_task_output(
            text,
            lambda value: ["a", "b"] if value == "candidate response" else None,
            lambda value: "final" if value == "final response" else None,
        )
        self.assertEqual(parsed["candidates"], ["a", "b"])
        self.assertEqual(parsed["parsed"], "final")

    def test_fused_output_rejects_malformed_envelopes(self):
        valid = format_fused_sft_output(
            "candidate thinking",
            "candidate response",
            "post-edit thinking",
            "post-edit response",
        )
        malformed = [
            valid.replace("<thinking>", "", 1),
            valid + "\n<thinking>extra</thinking>",
            valid.replace("candidate thinking", "", 1),
            "prefix\n" + valid,
            valid.replace(
                "<thinking>\ncandidate thinking\n</thinking>\n"
                "<response>\ncandidate response\n</response>",
                "<response>\ncandidate response\n</response>\n"
                "<thinking>\ncandidate thinking\n</thinking>",
            ),
        ]
        for value in malformed:
            with self.subTest(value=value):
                self.assertIsNone(parse_fused_sft_output(value))
        self.assertIsNone(
            parse_fused_task_output(valid, lambda _: None, lambda _: "final")
        )
        self.assertIsNone(
            parse_fused_task_output(valid, lambda _: ["a", "b"], lambda _: None)
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            format_fused_sft_output(
                "bad <thinking>", "candidates", "post-edit", "final"
            )

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_legacy_row_without_mode_metadata_remains_independent(
        self, from_pretrained
    ):
        class Tokenizer:
            @staticmethod
            def apply_chat_template(*args, **kwargs):
                return [0] * 32

        from_pretrained.return_value = Tokenizer()
        candidates = ["你好", "您好", "嗨", "你好呀"]
        responses = [
            f"<final_translation>{candidate}</final_translation>"
            for candidate in candidates
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "legacy.parquet"
            output_base = Path(temp_dir) / "train.parquet"
            pd.DataFrame({
                "src_text": ["Hello"],
                "src_lang": ["en"],
                "trg_lang": ["zh"],
                "gpe_stage1_thinking": [["Think"] * 4],
                "gpe_stage1_responses": [responses],
                "gpe_stage1_translations": [candidates],
                "gpe_stage2_thinking": ["Choose the formal candidate."],
                "gpe_stage2_response": ["```translation\n您好\n```"],
                "gpe_translation": ["您好"],
                "gpe_parser_valid": [True],
            }).to_parquet(input_path, index=False)

            prepare_gpe(
                str(input_path),
                str(output_base),
                tokenizer_path="unused",
            )
            stage1 = pd.read_parquet(
                Path(temp_dir) / "train.direct_mt.parquet"
            )
            post_edit = pd.read_parquet(
                Path(temp_dir) / "train.group_post_edit.parquet"
            )

        self.assertEqual(len(stage1), 4)
        self.assertEqual(set(stage1["task"]), {"direct_mt"})
        self.assertEqual(len(post_edit), 1)

    def test_gpe_output_paths_are_split_by_default(self):
        direct, post_edit = _output_paths("/tmp/train.parquet", None, None)
        self.assertEqual(str(direct), "/tmp/train.direct_mt.parquet")
        self.assertEqual(
            str(post_edit), "/tmp/train.group_post_edit.parquet"
        )

    def test_gpe_output_paths_can_be_overridden(self):
        direct, post_edit = _output_paths(
            "/tmp/unused.parquet", "/tmp/direct.parquet", "/tmp/gpe.parquet"
        )
        self.assertEqual(str(direct), "/tmp/direct.parquet")
        self.assertEqual(str(post_edit), "/tmp/gpe.parquet")
        with self.assertRaises(ValueError):
            _output_paths("/tmp/x", "/tmp/same", "/tmp/same")

    def test_round_trip_and_inner_parser(self):
        text = format_sft_output(
            "Check fidelity.",
            "<final_translation>Hello</final_translation>",
        )
        envelope = parse_sft_output(text)
        self.assertEqual(envelope.thinking, "Check fidelity.")
        parsed = parse_task_output(text, extract_final_translation)
        self.assertEqual(parsed["parsed"], "Hello")

    def test_rejects_invalid_envelopes(self):
        invalid = [
            "<response>x</response>",
            "<thinking></thinking><response>x</response>",
            "<thinking>x</thinking><response></response>",
            "prefix<thinking>x</thinking><response>y</response>",
            "<thinking>x</thinking><response>y</response>suffix",
            "<thinking>x</thinking><thinking>z</thinking><response>y</response>",
            "<response>y</response><thinking>x</thinking>",
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(parse_sft_output(value))

    def test_reserved_tags_cannot_appear_inside_sections(self):
        with self.assertRaises(ValueError):
            format_sft_output("bad </thinking>", "answer")

    def test_followup_uses_history_without_embedding_analysis(self):
        prompt = build_scd_followup_prompt("Chinese", "English")
        self.assertIn("segment candidates above", prompt)
        self.assertNotIn("<divergent_analysis>", prompt)
        self.assertNotIn("<thinking>", prompt)

    def test_training_prompts_are_short_and_omit_format_requirements(self):
        prompts = [
            build_sft_direct_prompt("en", "zh", "Hello"),
            build_sft_flash_gpe_candidate_prompt("en", "zh", "Hello", 4),
            build_sft_fused_flash_gpe_prompt("en", "zh", "Hello", 4),
            build_sft_gpe_prompt("en", "zh", "Hello", ["你好", "您好"]),
            build_scd_stage1_prompt("en", "zh", "Hello"),
            build_scd_followup_prompt("en", "zh"),
        ]
        for prompt in prompts:
            self.assertNotIn("<thinking>", prompt)
            self.assertNotIn("<response>", prompt)
            self.assertNotIn("JSON", prompt)
            self.assertNotIn("Output format", prompt)


if __name__ == "__main__":
    unittest.main()
