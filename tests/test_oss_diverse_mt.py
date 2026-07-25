import unittest
from unittest.mock import patch

from inference.oss_diverse_mt_prompts import (
    build_convergent_prompt,
    build_direct_prompt,
    build_divergent_prompt,
)
from inference.run_oss_diverse_mt import (
    extract_codeblock_response,
    extract_final_translation,
    extract_json_object,
    run_pipeline,
    validate_divergent_result,
)


class OssDiverseMtTest(unittest.TestCase):
    def test_divergent_json_extraction_and_validation(self):
        response = '''```json
{"source_analysis":"x","segments":[{"segment_id":1,"source_span":"hello","analysis":"x","candidates":[{"translation":"你好","angle":"neutral"}]}]}
```'''
        parsed = extract_json_object(response)
        self.assertEqual(validate_divergent_result(parsed), parsed)


    def test_divergent_validation_rejects_bad_segment_ids(self):
        parsed = {
            "source_analysis": "x",
            "segments": [{
                "segment_id": 2,
                "source_span": "hello",
                "analysis": "x",
                "candidates": [{"translation": "你好", "angle": "neutral"}],
            }],
        }
        self.assertIsNone(validate_divergent_result(parsed))


    def test_final_translation_extraction(self):
        self.assertEqual(
            extract_final_translation("<final_translation> 你好 </final_translation>"),
            "你好",
        )
        self.assertEqual(extract_final_translation("你好"), "你好")
        self.assertEqual(extract_final_translation("```zh\n你好\n```"), "你好")


    def test_prompts_define_distinct_stage_contracts(self):
        divergent = build_divergent_prompt("en", "zh", "Hello")
        convergent = build_convergent_prompt(
            "en", "zh", "Hello", {"source_analysis": "x", "segments": []}
        )
        direct = build_direct_prompt("en", "zh", "Hello")
        self.assertIn('"segments"', divergent)
        self.assertIn("Do not choose or compose a final translation", divergent)
        self.assertIn("<divergent_analysis>", convergent)
        self.assertNotIn("<divergent_analysis>", direct)
        self.assertIn("<final_translation>", convergent)
        self.assertIn("<final_translation>", direct)

    def test_codeblock_prompt_and_extraction_are_free_form(self):
        prompt = build_divergent_prompt(
            "en", "zh", "Hello", prompt_type="codeblock"
        )
        response = """Step-by-step analysis
The greeting is simple, but register can vary.

```markdown
Segment: Hello
- 你好 — neutral
- 您好 — polite
- 嗨 — casual
```"""
        self.assertIn("any clear natural-language format", prompt)
        self.assertNotIn('"segments"', prompt)
        self.assertEqual(extract_codeblock_response(response), response)
        self.assertIsNone(extract_codeblock_response("Segment: Hello"))

    @patch("inference.run_oss_diverse_mt.run_convergent_stage")
    @patch("inference.run_oss_diverse_mt.run_divergent_stage")
    def test_diverse_pipeline_does_not_run_or_return_direct(
        self, divergent_stage, convergent_stage
    ):
        divergent_stage.return_value = {
            "analyses": [{"source_analysis": "x", "segments": []}],
            "responses": ["stage 1"],
            "thinking": [None],
        }
        convergent_stage.return_value = {
            "translations": ["你好"],
            "responses": ["stage 2"],
            "thinking": [None],
        }
        result = run_pipeline(
            ["hello"], ["en"], ["zh"], model=object(), model_path="unused"
        )
        self.assertEqual(set(result), {"divergent", "convergent"})
        self.assertNotIn("direct", result)


if __name__ == "__main__":
    unittest.main()
