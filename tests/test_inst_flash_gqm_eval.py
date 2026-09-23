import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from eval.run_inst_mt_flash_gqm_eval import main


class InstFlashGqmEvalTest(unittest.TestCase):
    @patch("eval.run_inst_mt_flash_gqm_eval.log_results_to_wandb")
    @patch("eval.run_inst_mt_flash_gqm_eval._split_scores_by_data_id")
    @patch("eval.run_inst_mt_flash_gqm_eval.run_bleurt_eval")
    @patch("eval.run_inst_mt_flash_gqm_eval._release_vllm_model")
    @patch("eval.run_inst_mt_flash_gqm_eval.run_pipeline")
    @patch("eval.run_inst_mt_flash_gqm_eval.init_inst_model")
    @patch("eval.run_inst_mt_flash_gqm_eval._load_datasets")
    def test_eval_uses_selected_translations_and_logs_gqm_method(
        self,
        load_datasets,
        init_model,
        pipeline,
        _release,
        bleurt,
        split_scores,
        log_results,
    ):
        frame = pd.DataFrame({
            "src_text": ["one", "two"],
            "src_lang": ["en", "en"],
            "trg_lang": ["zh", "zh"],
            "trg_text": ["一", "二"],
        })
        load_datasets.return_value = (
            frame,
            {"toy": (0, 2)},
            {"toy": "en-zh"},
            {"toy": frame},
        )
        init_model.return_value = SimpleNamespace(model=object())
        pipeline.return_value = {
            "usable_candidate_counts": [2, 2],
            "gqm": {"translations": ["一", None]},
        }
        bleurt.return_value = [0.9, 0.0]
        split_scores.return_value = {
            "toy": {
                "avg": 0.45,
                "none_count": 0,
                "per_item_avgs": [0.9, 0.0],
            }
        }

        main(
            data_id=("toy",),
            model_path="model",
            model_name="name",
            metrics=["bleurt"],
        )

        self.assertEqual(
            bleurt.call_args.args[1], ["一", "Translation Failed."]
        )
        self.assertEqual(log_results.call_args.args[1]["method"], "inst_flash_gqm")
        self.assertEqual(
            log_results.call_args.args[1]["tie_break"], "first_maximum"
        )


if __name__ == "__main__":
    unittest.main()
