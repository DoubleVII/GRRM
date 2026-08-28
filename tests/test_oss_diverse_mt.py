import unittest
from unittest.mock import patch

import pandas as pd

from eval.run_oss_diverse_mt_eval import (
    _diversity_stats,
    _prompt_variant,
    _score_translations,
    _summary_for_indices,
)
from inference.oss_diverse_mt_prompts import (
    build_convergent_prompt,
    build_direct_prompt,
    build_divergent_prompt,
)
from inference.run_oss_diverse_mt import (
    extract_codeblock_response,
    extract_final_translation,
    extract_json_object,
    normalize_bool,
    run_pipeline,
    validate_decision_points_result,
    validate_divergent_result,
)


class OssDiverseMtTest(unittest.TestCase):
    def test_semantic_units_prompt_allows_overlap_and_complete_spans(self):
        prompt = build_divergent_prompt(
            "en",
            "zh",
            "Not everyone who applied was accepted.",
            prompt_type="semantic_units",
        )
        self.assertIn("parent-child units are encouraged", prompt)
        self.assertIn("parent unit should preserve the complete meaning", prompt)
        self.assertIn("child candidates preserve local alternatives", prompt)
        self.assertIn("single word or term", prompt)
        self.assertIn("to a complete sentence", prompt)
        self.assertIn("negation or quantifier scope", prompt)
        self.assertIn("Use semantically complete parent units", prompt)
        self.assertNotIn("non-overlapping translation units", prompt)
        self.assertIn('"segments"', prompt)

    def test_semantic_units_uses_standard_polished_convergent_prompt(self):
        prompt = build_convergent_prompt(
            "en",
            "zh",
            "Not everyone agreed.",
            {"source_analysis": "x", "segments": []},
            prompt_type="semantic_units",
            polish=True,
        )
        self.assertIn("mandatory whole-text polish pass", prompt)
        self.assertIn("freely rewriting across segment boundaries", prompt)
        self.assertIn("<divergent_analysis>", prompt)

    def test_semantic_units_statistics_count_containment_overlaps(self):
        analyses = [{
            "source_analysis": "x",
            "segments": [{
                "segment_id": 1,
                "source_span": "Not everyone agreed",
                "analysis": "scope",
                "candidates": [{"translation": "并非所有人都同意", "angle": "x"}],
            }, {
                "segment_id": 2,
                "source_span": "everyone",
                "analysis": "quantifier",
                "candidates": [{"translation": "所有人", "angle": "x"}],
            }],
        }]
        stats = _diversity_stats(analyses, "semantic_units")
        self.assertEqual(stats["mean_segments_per_item"], 2)
        self.assertEqual(stats["mean_containment_overlaps_per_item"], 1)


    def test_decision_points_prompt_is_selective_and_non_exhaustive(self):
        prompt = build_divergent_prompt(
            "en",
            "zh",
            "Not everyone had the same advantages.",
            prompt_type="decision_points",
            max_decision_points=4,
        )
        self.assertIn("do not divide it into exhaustive translation segments", prompt)
        self.assertIn("Return zero decision points", prompt)
        self.assertIn("limit of 4 is a hard ceiling", prompt)
        self.assertIn("Normally return 0 to 2 decision points", prompt)
        self.assertIn("$10 billion is 100亿美元, not 10亿美元", prompt)
        self.assertIn('"global_constraints"', prompt)
        self.assertIn('"decision_points"', prompt)
        self.assertNotIn('"segments"', prompt)

    def test_decision_points_validation_accepts_empty_and_structured_results(self):
        empty = {
            "source_analysis": "Simple source.",
            "global_constraints": [],
            "decision_points": [],
        }
        self.assertEqual(validate_decision_points_result(empty, 4), empty)

        structured = {
            "source_analysis": "Preserve negation scope.",
            "global_constraints": ["Not all people had the same advantages."],
            "decision_points": [{
                "decision_point_id": 1,
                "source_span": "advantages",
                "issue_type": "word_sense",
                "analysis": "Context means favorable conditions.",
                "candidates": [
                    {"translation": "优越条件", "angle": "contextual"},
                    {"translation": "优势", "angle": "concise"},
                ],
            }],
        }
        self.assertEqual(validate_decision_points_result(structured, 4), structured)
        structured["decision_points"][0]["decision_point_id"] = 2
        self.assertIsNone(validate_decision_points_result(structured, 4))

    def test_decision_points_validation_rejects_confidence_and_excess_points(self):
        point = {
            "decision_point_id": 1,
            "source_span": "term",
            "issue_type": "terminology",
            "analysis": "ambiguous term",
            "candidates": [{
                "translation": "术语",
                "angle": "standard",
                "confidence": "high",
            }],
        }
        value = {
            "source_analysis": "x",
            "global_constraints": [],
            "decision_points": [point],
        }
        self.assertIsNone(validate_decision_points_result(value, 4))
        point["candidates"][0].pop("confidence")
        value["decision_points"] = [point, {**point, "decision_point_id": 2}]
        self.assertIsNone(validate_decision_points_result(value, 1))

    def test_decision_points_convergent_prompt_drafts_from_full_source_first(self):
        context = {
            "source_analysis": "x",
            "global_constraints": ["Preserve negation scope."],
            "decision_points": [],
        }
        prompt = build_convergent_prompt(
            "en",
            "zh",
            "Not everyone agreed.",
            context,
            prompt_type="decision_points",
            polish=True,
        )
        self.assertIn("independently construct", prompt)
        self.assertIn("must never be treated as pieces to concatenate", prompt)
        self.assertIn("negation and quantifier scope", prompt)
        self.assertIn("Independently recompute numeric magnitudes", prompt)
        self.assertIn("Keep the independent draft", prompt)
        self.assertIn("<decision_point_analysis>", prompt)
        self.assertNotIn("<divergent_analysis>", prompt)

    def test_decision_points_statistics(self):
        analyses = [{
            "source_analysis": "x",
            "global_constraints": ["constraint"],
            "decision_points": [{
                "decision_point_id": 1,
                "source_span": "term",
                "issue_type": "terminology",
                "analysis": "x",
                "candidates": [
                    {"translation": "A", "angle": "x"},
                    {"translation": "A", "angle": "duplicate"},
                    {"translation": "B", "angle": "y"},
                ],
            }],
        }, {
            "source_analysis": "simple",
            "global_constraints": [],
            "decision_points": [],
        }]
        stats = _diversity_stats(analyses, "decision_points")
        self.assertEqual(stats["mean_decision_points_per_item"], 0.5)
        self.assertEqual(stats["items_without_decision_points"], 1)
        self.assertEqual(stats["mean_candidates_per_decision_point"], 3)
        self.assertAlmostEqual(stats["exact_duplicate_candidate_rate"], 1 / 3)

    def test_decision_points_rejects_candidate_confidence(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            build_divergent_prompt(
                "en",
                "zh",
                "Hello",
                prompt_type="decision_points",
                candidate_confidence=True,
            )

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

    def test_candidate_confidence_prompt_and_validation(self):
        prompt = build_divergent_prompt(
            "en", "zh", "Hello", candidate_confidence=True
        )
        self.assertIn('"confidence": "high"', prompt)
        self.assertIn("Confidence is not a diversity dimension", prompt)
        self.assertIn("Most candidates for an unambiguous unit", prompt)

        candidate = {"translation": "你好", "angle": "neutral"}
        parsed = {
            "source_analysis": "x",
            "segments": [{
                "segment_id": 1,
                "source_span": "Hello",
                "analysis": "greeting",
                "candidates": [candidate],
            }],
        }
        self.assertIsNone(
            validate_divergent_result(parsed, candidate_confidence=True)
        )
        candidate["confidence"] = "certain"
        self.assertIsNone(
            validate_divergent_result(parsed, candidate_confidence=True)
        )
        candidate["confidence"] = "high"
        self.assertEqual(
            validate_divergent_result(parsed, candidate_confidence=True), parsed
        )


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

    def test_polish_and_confidence_are_independent_prompt_switches(self):
        context = {"source_analysis": "x", "segments": []}
        plain = build_convergent_prompt(
            "en", "zh", "Hello", context, polish=False
        )
        combined = build_convergent_prompt(
            "en",
            "zh",
            "Hello",
            context,
            polish=True,
            candidate_confidence=True,
        )
        self.assertNotIn("mandatory whole-text polish pass", plain)
        self.assertNotIn("calibrated aid", plain)
        self.assertIn("mandatory whole-text polish pass", combined)
        self.assertIn("calibrated aid", combined)
        self.assertIn("Reject even a high-confidence candidate", combined)

    def test_prompt_variant_names_all_switches(self):
        self.assertEqual(
            _prompt_variant("json", True, True), "json.polish.confidence"
        )
        self.assertEqual(
            _prompt_variant("codeblock", False, False),
            "codeblock.no-polish.no-confidence",
        )
        self.assertEqual(
            _prompt_variant("decision_points", True, False, 4),
            "decision_points.polish.no-confidence.dp4",
        )

    def test_cli_boolean_normalization(self):
        self.assertTrue(normalize_bool("true", "flag"))
        self.assertFalse(normalize_bool("false", "flag"))
        self.assertFalse(normalize_bool(False, "flag"))
        with self.assertRaisesRegex(ValueError, "flag must be a boolean"):
            normalize_bool("sometimes", "flag")

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
            ["hello"],
            ["en"],
            ["zh"],
            model=object(),
            model_path="unused",
            polish=False,
            candidate_confidence=True,
        )
        self.assertEqual(set(result), {"divergent", "convergent"})
        self.assertNotIn("direct", result)
        self.assertTrue(divergent_stage.call_args.kwargs["candidate_confidence"])
        self.assertFalse(convergent_stage.call_args.kwargs["polish"])
        self.assertTrue(convergent_stage.call_args.kwargs["candidate_confidence"])

    @patch("inference.run_oss_diverse_mt.run_convergent_stage")
    @patch("inference.run_oss_diverse_mt.run_divergent_stage")
    def test_decision_points_pipeline_propagates_prompt_configuration(
        self, divergent_stage, convergent_stage
    ):
        divergent_stage.return_value = {
            "analyses": [{
                "source_analysis": "x",
                "global_constraints": [],
                "decision_points": [],
            }],
            "responses": ["stage 1"],
            "thinking": [None],
        }
        convergent_stage.return_value = {
            "translations": ["你好"],
            "responses": ["stage 2"],
            "thinking": [None],
        }

        run_pipeline(
            ["hello"],
            ["en"],
            ["zh"],
            model=object(),
            model_path="unused",
            prompt_type="decision_points",
            candidate_confidence=False,
            max_decision_points=4,
        )

        self.assertEqual(
            divergent_stage.call_args.kwargs["prompt_type"], "decision_points"
        )
        self.assertEqual(
            divergent_stage.call_args.kwargs["max_decision_points"], 4
        )
        self.assertEqual(
            convergent_stage.call_args.kwargs["prompt_type"], "decision_points"
        )


if __name__ == "__main__":
    unittest.main()
