import unittest
from unittest.mock import patch

import pandas as pd

from eval.run_oss_diverse_mt_eval import _score_translations, _summary_for_indices
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
        self.assertIn("mandatory whole-text polish pass", convergent)
        self.assertIn("rewriting across segment boundaries", convergent)
        self.assertIn("check the full translation against the source again", convergent)
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

    @patch("eval.run_oss_diverse_mt_eval.run_oss_sqm")
    def test_evaluation_runs_repeat_scoring_and_average_per_item(self, run_oss_sqm):
        frame = pd.DataFrame(
            {
                "src_text": ["one", "two"],
                "trg_text": ["一", "二"],
                "src_lang": ["en", "en"],
                "trg_lang": ["zh", "zh"],
            }
        )
        run_oss_sqm.return_value = {
            "scores": [80, 90, 100, 70, 90, 80],
            "response": [f"response-{index}" for index in range(6)],
        }

        result = _score_translations(
            frame, ["一", "二"], object(), "unused", runs=3
        )

        self.assertEqual(
            run_oss_sqm.call_args.kwargs["src_list"],
            ["one", "two", "one", "two", "one", "two"],
        )
        self.assertEqual(
            result["scores_by_run"], [[80, 90], [100, 70], [90, 80]]
        )
        self.assertEqual(result["scores"], [90, 80])

        summary = _summary_for_indices(
            [0, 1], result["scores"], result["scores_by_run"], ["一", "二"]
        )
        self.assertEqual(summary["run_means"], [85, 85, 85])
        self.assertEqual(summary["two_stage_mean"], 85)
        self.assertEqual(summary["evaluation_runs"], 3)

    def test_evaluation_runs_must_be_positive(self):
        frame = pd.DataFrame(
            {"src_text": [], "trg_text": [], "src_lang": [], "trg_lang": []}
        )
        with self.assertRaisesRegex(ValueError, "runs must be at least 1"):
            _score_translations(frame, [], object(), "unused", runs=0)

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
