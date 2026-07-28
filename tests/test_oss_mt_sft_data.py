import unittest

from data.oss_mt_sft_data_utils import (
    parse_scd_stage1_response,
    validate_gpe_record,
    validate_scd_record,
)


SCD_RESPONSE = """{
  "source_analysis": "A greeting.",
  "segments": [
    {
      "segment_id": 1,
      "source_span": "Hello",
      "analysis": "A conventional greeting.",
      "candidates": [
        {"translation": "你好", "angle": "neutral"}
      ]
    }
  ]
}"""


class OssMtSftDataTest(unittest.TestCase):
    def test_scd_requires_both_parser_valid_turns_and_thinking(self):
        analysis = parse_scd_stage1_response(
            SCD_RESPONSE, prompt_type="json", candidate_confidence=False
        )
        kwargs = {
            "stage1_response": SCD_RESPONSE,
            "stage1_thinking": "I should preserve the greeting.",
            "stage1_analysis": analysis,
            "stage2_response": "<final_translation>你好</final_translation>",
            "stage2_thinking": "The neutral candidate is appropriate.",
            "stage2_translation": "你好",
            "prompt_type": "json",
            "candidate_confidence": False,
        }
        self.assertTrue(validate_scd_record(**kwargs))
        self.assertFalse(validate_scd_record(**{**kwargs, "stage1_thinking": None}))
        self.assertFalse(
            validate_scd_record(**{**kwargs, "stage2_translation": "您好"})
        )

    def test_scd_rejects_invalid_json(self):
        self.assertIsNone(
            parse_scd_stage1_response(
                '{"source_analysis": "missing segments"}',
                prompt_type="json",
                candidate_confidence=False,
            )
        )

    def test_gpe_validates_every_direct_sample_and_post_edit(self):
        kwargs = {
            "candidate_responses": [
                "<final_translation>你好</final_translation>",
                "<final_translation>您好</final_translation>",
            ],
            "candidate_thinking": ["Use neutral register.", "Use formal register."],
            "candidate_translations": ["你好", "您好"],
            "post_edit_response": "```translation\n您好\n```",
            "post_edit_thinking": "The formal candidate fits best.",
            "post_edit_translation": "您好",
            "sampling_n": 2,
        }
        self.assertTrue(validate_gpe_record(**kwargs))
        self.assertFalse(
            validate_gpe_record(
                **{**kwargs, "candidate_responses": kwargs["candidate_responses"][:1]}
            )
        )
        self.assertFalse(
            validate_gpe_record(**{**kwargs, "post_edit_response": "no code block"})
        )


if __name__ == "__main__":
    unittest.main()
