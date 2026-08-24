import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from eval.run_oss_flash_gpe_mt_eval import main as run_eval
from inference.oss_flash_gpe_prompts import (
    build_candidate_prompt,
    build_post_edit_prompt,
    validate_prompt_type,
)
from inference.run_oss_flash_gpe_mt import (
    extract_candidate_response,
    main as run_inference,
    run_candidate_generation_stage,
    run_pipeline,
)


class OssFlashGpeMtTest(unittest.TestCase):
    def test_public_api_only_accepts_flash_gpe_parameters(self):
        parameters = inspect.signature(run_inference).parameters
        self.assertIn("max_candidates", parameters)
        self.assertIn("prompt_type", parameters)
        self.assertNotIn("sampling_n", parameters)
        self.assertNotIn("candidate_generation", parameters)

    def test_post_edit_prompt_supports_more_than_eight_candidates(self):
        prompt = build_post_edit_prompt(
            "en", "zh", "source", [f"candidate-{i}" for i in range(12)]
        )
        self.assertIn("Translation 12", prompt)

    def test_prompt_types_have_distinct_count_guidance(self):
        fixed_4 = build_candidate_prompt("en", "zh", "source", 4, "fixed_4")
        adaptive = build_candidate_prompt("en", "zh", "source", 10, "adaptive")
        fixed_16 = build_candidate_prompt("en", "zh", "source", 16, "fixed_16")
        self.assertIn("exactly 4 complete translation candidates", fixed_4)
        self.assertIn("never more than 10", adaptive)
        self.assertNotIn("prefer 8 candidates", adaptive)
        self.assertIn("prefer 8 candidates", fixed_16)
        self.assertIn("12 to 16 candidates", fixed_16)
        validate_prompt_type("fixed_4", 4)
        validate_prompt_type("adaptive", 6)
        validate_prompt_type("fixed_16", 16)
        with self.assertRaisesRegex(ValueError, "max_candidates=4"):
            validate_prompt_type("fixed_4", 8)
        with self.assertRaisesRegex(ValueError, "max_candidates=16"):
            validate_prompt_type("fixed_16", 8)

    def test_parser_validates_count_content_and_uniqueness(self):
        response = json.dumps({"translations": ["你好", "您好", "嗨"]})
        self.assertEqual(
            extract_candidate_response(response, 16),
            ["你好", "您好", "嗨"],
        )
        self.assertIsNone(extract_candidate_response(response, 2))
        self.assertIsNone(
            extract_candidate_response(response, 4, exact_count=True)
        )
        duplicate = json.dumps(
            {"translations": ["Hello world", " hello  world "]}
        )
        self.assertIsNone(extract_candidate_response(duplicate, 2))

    def test_markdown_parser_supports_multiline_candidates(self):
        response = (
            "# Candidate 1\n\n第一行\n第二行\n\n"
            "# Candidate 2\n\n另一条翻译"
        )
        self.assertEqual(
            extract_candidate_response(response, 2, exact_count=True),
            ["第一行\n第二行", "另一条翻译"],
        )
        self.assertIsNone(
            extract_candidate_response(
                "# Candidate 1\nA\n# Candidate 3\nC", 2, exact_count=True
            )
        )

    @patch("inference.run_oss_flash_gpe_mt._generate_with_retries")
    @patch("inference.run_oss_flash_gpe_mt._prepare_inputs")
    @patch("inference.run_oss_flash_gpe_mt.load_encoding")
    def test_candidate_stage_uses_one_request_per_source(
        self, load_encoding, prepare_inputs, generate
    ):
        load_encoding.return_value = object()
        prepare_inputs.side_effect = lambda prompts, *_: prompts
        generate.return_value = [
            {"parsed": ["a1", "a2"], "response": "ra", "thinking": "ta"},
            {"parsed": ["b1", "b2"], "response": "rb", "thinking": "tb"},
        ]
        result = run_candidate_generation_stage(
            ["a", "b"],
            ["en", "zh"],
            ["zh", "en"],
            max_candidates=3,
            prompt_type="adaptive",
            model=object(),
        )
        self.assertEqual(len(generate.call_args.args[1]), 2)
        self.assertEqual(result["translations"], [["a1", "a2"], ["b1", "b2"]])
        self.assertEqual(result["responses"], ["ra", "rb"])

    @patch("inference.run_oss_flash_gpe_mt.run_group_post_edit")
    @patch("inference.run_oss_flash_gpe_mt.run_candidate_generation_stage")
    def test_pipeline_skips_items_with_too_few_candidates(
        self, candidate_stage, post_edit
    ):
        candidate_stage.return_value = {
            "prompts": ["pa", "pb"],
            "translations": [["a1", "a2"], ["only"]],
            "responses": ["ra", "rb"],
            "thinking": ["ta", "tb"],
        }
        post_edit.return_value = {
            "post_edit_mt": ["final-a"],
            "response": ["response-a"],
            "thinking": ["thinking-a"],
        }
        result = run_pipeline(
            ["a", "b"],
            ["en", "en"],
            ["zh", "zh"],
            model=object(),
            model_path="unused",
            max_candidates=3,
            prompt_type="adaptive",
        )
        self.assertEqual(post_edit.call_args.kwargs["src_list"], ["a"])
        self.assertEqual(result["usable_candidate_counts"], [2, 1])
        self.assertEqual(
            result["post_edit"]["translations"], ["final-a", None]
        )

    @patch("eval.run_oss_flash_gpe_mt_eval._score_translations")
    @patch("eval.run_oss_flash_gpe_mt_eval.init_oss_model")
    @patch("eval.run_oss_flash_gpe_mt_eval._load_data")
    @patch("eval.run_oss_flash_gpe_mt_eval.run_pipeline")
    def test_eval_uses_flash_gpe_schema(
        self, pipeline, load_data, init_model, score_translations
    ):
        load_data.return_value = pd.DataFrame({
            "data_id": ["toy"],
            "source_index": [0],
            "src_lang": ["en"],
            "trg_lang": ["zh"],
            "src_text": ["Hello"],
            "trg_text": ["你好"],
        })
        init_model.return_value = object()
        pipeline.return_value = {
            "candidate_generation": {
                "translations": [["你好", "您好", "嗨", "你好呀"]],
                "responses": ["joint"],
                "thinking": ["candidate-thinking"],
            },
            "usable_candidate_counts": [4],
            "post_edit": {
                "translations": ["您好"],
                "responses": ["final-response"],
                "thinking": ["final-thinking"],
            },
        }
        score_translations.return_value = {
            "scores": [90.0],
            "scores_by_run": [[90.0]],
            "responses_by_run": [["eval"]],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "result.json"
            run_eval(
                data_id="toy",
                output_path=str(output_path),
                max_samples=1,
                max_candidates=4,
                prompt_type="fixed_4",
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["method"], "flash_gpe")
        self.assertEqual(payload["summary"]["overall"]["flash_gpe_mean"], 90.0)
        self.assertEqual(payload["summary"]["overall"]["candidate_count_mean"], 4)
        self.assertEqual(payload["items"][0]["flash_gpe_translation"], "您好")


if __name__ == "__main__":
    unittest.main()
