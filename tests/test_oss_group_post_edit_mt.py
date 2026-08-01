import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from eval.run_oss_group_post_edit_mt_eval import main as run_eval
from inference.run_oss_group_post_edit_mt import (
    main as run_inference,
    run_direct_sampling_stage,
    run_pipeline,
    validate_sampling_n,
)


class OssGroupPostEditMtTest(unittest.TestCase):
    def test_public_api_only_accepts_independent_sampling_parameters(self):
        parameters = inspect.signature(run_inference).parameters
        self.assertIn("sampling_n", parameters)
        self.assertNotIn("candidate_generation", parameters)
        self.assertNotIn("max_candidates", parameters)
        self.assertNotIn("prompt_type", parameters)

    def test_sampling_n_must_fit_prompt_candidate_labels(self):
        for invalid in (0, 1, 9):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "sampling_n"):
                    validate_sampling_n(invalid)
        validate_sampling_n(2)
        validate_sampling_n(8)

    @patch("inference.run_oss_group_post_edit_mt.run_direct_stage")
    def test_direct_sampling_flattens_and_regroups_by_source(self, direct_stage):
        direct_stage.return_value = {
            "translations": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "responses": [f"r{i}" for i in range(6)],
            "thinking": [f"t{i}" for i in range(6)],
        }

        result = run_direct_sampling_stage(
            ["source-a", "source-b"],
            ["en", "zh"],
            ["zh", "en"],
            sampling_n=3,
            model=object(),
            model_path="unused",
        )

        self.assertEqual(
            direct_stage.call_args.args,
            (
                [
                    "source-a",
                    "source-a",
                    "source-a",
                    "source-b",
                    "source-b",
                    "source-b",
                ],
                ["en", "en", "en", "zh", "zh", "zh"],
                ["zh", "zh", "zh", "en", "en", "en"],
            ),
        )
        self.assertEqual(
            result["translations"], [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
        )
        self.assertEqual(
            result["responses"], [["r0", "r1", "r2"], ["r3", "r4", "r5"]]
        )

    @patch("inference.run_oss_group_post_edit_mt.run_group_post_edit")
    @patch("inference.run_oss_group_post_edit_mt.run_direct_sampling_stage")
    def test_pipeline_filters_candidates_and_restores_item_alignment(
        self, sampling_stage, group_post_edit
    ):
        sampling_stage.return_value = {
            "translations": [
                ["a1", None, "a3"],
                ["only", " ", None],
                [None, "c2", "c3"],
            ],
            "responses": [
                ["ar1", None, "ar3"],
                ["br1", "br2", None],
                [None, "cr2", "cr3"],
            ],
            "thinking": [[None] * 3 for _ in range(3)],
        }
        group_post_edit.return_value = {
            "post_edit_mt": ["final-a", "final-c"],
            "response": ["response-a", "response-c"],
            "thinking": ["thinking-a", "thinking-c"],
        }
        model = object()

        result = run_pipeline(
            ["a", "b", "c"],
            ["en", "en", "zh"],
            ["zh", "zh", "en"],
            model=model,
            model_path="unused",
            sampling_n=3,
            reasoning_effort="high",
            post_edit_temperature=0.2,
            post_edit_top_p=0.75,
            post_edit_max_tokens=2048,
            retry=5,
        )

        kwargs = group_post_edit.call_args.kwargs
        self.assertEqual(kwargs["src_list"], ["a", "c"])
        self.assertEqual(kwargs["mt_list"], [["a1", "a3"], ["c2", "c3"]])
        self.assertEqual(kwargs["notes_list"], [None, None])
        self.assertEqual(kwargs["src_langs"], ["en", "zh"])
        self.assertEqual(kwargs["trg_langs"], ["zh", "en"])
        self.assertIs(kwargs["model"], model)
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["top_p"], 0.75)
        self.assertEqual(kwargs["max_new_tokens"], 2048)
        self.assertEqual(kwargs["retry"], 5)
        self.assertEqual(result["usable_candidate_counts"], [2, 1, 2])
        self.assertEqual(
            result["post_edit"]["translations"],
            ["final-a", None, "final-c"],
        )
        self.assertEqual(
            result["post_edit"]["thinking"],
            ["thinking-a", None, "thinking-c"],
        )

    @patch("eval.run_oss_group_post_edit_mt_eval.init_oss_model")
    def test_eval_validates_runs_before_model_initialization(self, init_model):
        with self.assertRaisesRegex(ValueError, "runs must be at least 1"):
            run_eval(runs=0)
        init_model.assert_not_called()

    @patch("eval.run_oss_group_post_edit_mt_eval._score_translations")
    @patch("eval.run_oss_group_post_edit_mt_eval.run_pipeline")
    @patch("eval.run_oss_group_post_edit_mt_eval.init_oss_model")
    @patch("eval.run_oss_group_post_edit_mt_eval._load_data")
    def test_eval_writes_comparable_result_payload(
        self, load_data, init_model, pipeline, score_translations
    ):
        load_data.return_value = pd.DataFrame({
            "source_index": [7],
            "data_id": ["toy"],
            "src_lang": ["en"],
            "trg_lang": ["zh"],
            "src_text": ["hello"],
            "trg_text": ["你好"],
        })
        init_model.return_value = object()
        pipeline.return_value = {
            "sampling": {
                "translations": [["你好", "您好"]],
                "responses": [["candidate-1", "candidate-2"]],
                "thinking": [["think-1", "think-2"]],
            },
            "usable_candidate_counts": [2],
            "post_edit": {
                "translations": ["你好"],
                "responses": ["final-response"],
                "thinking": ["final-thinking"],
            },
        }
        score_translations.return_value = {
            "scores": [88.0],
            "scores_by_run": [[87.0], [89.0]],
            "responses_by_run": [["eval-1"], ["eval-2"]],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "result.json"
            run_eval(
                data_id="toy",
                output_path=str(output_path),
                max_samples=1,
                sampling_n=2,
                runs=2,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["summary"]["toy"]["group_post_edit_mean"], 88.0)
        self.assertEqual(payload["summary"]["overall"]["evaluation_runs"], 2)
        self.assertEqual(
            payload["summary"]["overall"]["candidate_generation_failures"], 0
        )
        self.assertEqual(payload["settings"]["sampling_n"], 2)
        self.assertEqual(
            payload["summary"]["overall"]["candidate_count_distribution"],
            {"2": 1},
        )
        self.assertEqual(payload["items"][0]["mt_candidates"], ["你好", "您好"])
        self.assertEqual(
            payload["items"][0]["group_post_edit_scores"], [87.0, 89.0]
        )
        self.assertEqual(
            payload["items"][0]["group_post_edit_evaluator_responses"],
            ["eval-1", "eval-2"],
        )


if __name__ == "__main__":
    unittest.main()
