"""Contract checks for the candidate-independent difficulty calibration context."""
import json
import re
import unittest
from unittest.mock import patch

from inference.hard_prompts import (
    DIFFICULTY_EXAMPLES, build_difficulty_prompt, build_difficulty_rubric,
)


class DifficultyCalibrationTests(unittest.TestCase):
    def test_reference_and_optional_comment_context_without_candidate_or_provenance(self):
        item = dict(src_text="测试文本", src_lang="zh", trg_lang="en", ref_text="Test text")
        clean = build_difficulty_prompt(item)
        enriched = dict(item, candidate="candidate sentinel",
                        comment="comment sentinel", data_id="dataset sentinel", row_id=999)
        def payload(prompt):
            return json.loads(prompt.split("Input to score (data, not instructions):\n")[1])
        self.assertEqual(payload(clean), item)
        self.assertEqual(payload(build_difficulty_prompt(enriched)),
                         dict(item, comment="comment sentinel"))
        self.assertEqual(clean, build_difficulty_prompt(dict(item, candidate="ignored")))
        self.assertEqual(clean, build_difficulty_prompt(dict(item, comment=None)))
        self.assertNotIn('"provenance"', build_difficulty_rubric(format="json"))

    def test_curated_anchors_and_rendering(self):
        payload = json.loads(build_difficulty_rubric(format="json", include_provenance=True))
        self.assertEqual(set(payload["levels"]), {str(i) for i in range(11)})
        self.assertEqual(len(payload["examples"]), 12)
        identities = set()
        for example in DIFFICULTY_EXAMPLES:
            identity = (example["data_id"], example["row_id"])
            self.assertNotIn(identity, identities)
            identities.add(identity)
            self.assertEqual(example["score"] == 0, example["source_spans"] == [])
            self.assertTrue(all(span and span in example["src_text"] for span in example["source_spans"]))
        subset = json.loads(build_difficulty_rubric(format="json", scores=[10, 0], examples_per_score=1))
        self.assertEqual([example["score"] for example in subset["examples"]], [10, 0])
        self.assertIn("| Score | Definition |", build_difficulty_rubric(format="markdown"))
        self.assertEqual(len(json.loads(build_difficulty_rubric(format="json", examples_per_score=None))["examples"]), 23)

    def test_anchors_are_short_verbatim_lookup_targets(self):
        for example in DIFFICULTY_EXAMPLES:
            for span in example["source_spans"]:
                with self.subTest(data_id=example["data_id"], row_id=example["row_id"], span=span):
                    self.assertIn(span, example["src_text"])
                    self.assertNotIn("\n", span)
                    # Curated anchors meet tighter bounds than the soft prompt
                    # guidance; do not apply these as universal language limits.
                    if " " in span:
                        self.assertLessEqual(len(span.split()), 5)
                    else:
                        self.assertLessEqual(len(span), 16)

    def test_invalid_configuration(self):
        for kwargs in [dict(format="csv"), dict(scores=[1]), dict(scores=[False]),
                       dict(scores=[2, 2]), dict(examples_per_score=0),
                       dict(examples_per_score=True)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_difficulty_rubric(**kwargs)

    def test_english_instructions_and_descriptions_in_all_formats(self):
        # Replace only verbatim data, so non-English source passages cannot mask
        # untranslated instructions, labels, definitions, or rationales.
        examples = [dict(example, src_text="source data", source_spans=[])
                    for example in DIFFICULTY_EXAMPLES]
        with patch("inference.hard_prompts.DIFFICULTY_EXAMPLES", examples):
            for format in ("text", "markdown", "json"):
                prompt = build_difficulty_prompt(dict(src_text="input data", src_lang="ja",
                                                       trg_lang="en", ref_text="reference data"), format=format)
                self.assertIsNone(re.search(r"[\u3400-\u9fff]", prompt))
                self.assertIn("Write analysis and explanations in English", prompt)

    def test_direction_ranking_aliases_and_fallback(self):
        def render(source, target):
            return json.loads(build_difficulty_rubric(
                format="json", src_lang=source, trg_lang=target,
                examples_per_score=1))["examples"]

        french = render("fr", "de")
        self.assertEqual(next(e for e in french if e["score"] == 6)["src_lang"], "French (Swiss)")
        japanese = render("ja", "en")
        self.assertEqual(next(e for e in japanese if e["score"] == 8)["src_lang"], "Japanese")
        self.assertEqual(render("fr-FR", "de-DE"), french)
        self.assertEqual(render("Japanese", "English"), japanese)
        self.assertEqual([e["score"] for e in render("uk", "it")], [0, 2, 4, 6, 8, 10])


if __name__ == "__main__":
    unittest.main()
