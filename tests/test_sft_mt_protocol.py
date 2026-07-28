import unittest

from inference.run_oss_diverse_mt import extract_final_translation
from inference.sft_mt_protocol import (
    add_output_instruction,
    build_scd_followup_prompt,
    format_sft_output,
    parse_sft_output,
    parse_task_output,
)


class SftMtProtocolTest(unittest.TestCase):
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
        self.assertIn("previous response", prompt)
        self.assertNotIn("<divergent_analysis>", prompt)
        self.assertIn("<thinking>", add_output_instruction("task"))


if __name__ == "__main__":
    unittest.main()
