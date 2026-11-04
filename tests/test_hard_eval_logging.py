"""Exercise real Fire entry points without loading GPU models."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from inference.hard_prompts import DIFFICULTY_RUBRIC_VERSION


class HardEvalLoggingTests(unittest.TestCase):
    def test_cli_keeps_full_records_in_artifacts_and_brief_results_in_logs(self):
        source = "source content sentinel"
        raw = "private generation sentinel" * 1000
        result = dict(analysis="analysis sentinel", score=5,
                      spans=[dict(source_span="content", score=5, explanation="explanation sentinel")])
        item = dict(data_id="tiny", row_id=0, src_text=source,
                    src_lang="en", trg_lang="zh", ref_text="参考译文",
                    status="ok", result=result)
        generated = dict(parsed=result, response=raw, thinking=raw,
                         raw_output=raw, attempts=[dict(response=raw)])
        for backend in ("inst", "oss"):
            with self.subTest(backend=backend), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "tiny.oss.difficulty_annotations.json").write_text(json.dumps(dict(
                    version=1, stage="difficulty_annotations",
                    prompt_version=DIFFICULTY_RUBRIC_VERSION, config={}, items=[item])))
                predictions, evaluation = root / "predictions", root / "evaluation"
                prediction_args = ["cli", "--data_id=tiny", f"--annotations_dir={root}",
                    f"--output_dir={predictions}", "--model_path=fake", "--runs=2"]
                output = io.StringIO()
                with redirect_stdout(output), patch.object(sys, "argv", prediction_args), patch(
                        "inference.hard_generation.generate", return_value=[generated, generated]):
                    runpy.run_module(f"inference.run_{backend}_hard_prediction", run_name="__main__")
                self.assertIn("Predictions saved:", output.getvalue())
                self.assertIn("'strict': 2", output.getvalue())
                self.assertNotIn(source, output.getvalue())
                self.assertNotIn(raw, output.getvalue())
                saved = json.loads((predictions / "tiny.direct.predictions.json").read_text())
                self.assertEqual(saved["items"][0]["generation"]["raw_output"], raw)

                eval_module = "eval.run_hard_eval" if backend == "inst" else "eval.run_oss_hard_eval"
                eval_args = ["cli", "--data_id=tiny", f"--annotations_dir={root}",
                    f"--predictions_dir={predictions}", f"--output_dir={evaluation}", "--model_path=fake"]
                output = io.StringIO()
                matched = dict(parsed=dict(analysis="judge analysis sentinel", pairs=[[0, 0]]),
                               raw_output=raw, thinking=raw)
                with redirect_stdout(output), patch.object(sys, "argv", eval_args), patch(
                        "eval.difficulty_eval_core.generate", return_value=[matched, matched]):
                    runpy.run_module(eval_module, run_name="__main__")
                log = output.getvalue()
                self.assertIn('"mae": 0.0', log)
                self.assertIn("Evaluation summary saved:", log)
                self.assertNotIn("by_run", log)
                self.assertNotIn(source, log)
                self.assertNotIn(raw, log)
                self.assertLess(len(log), 3000)
                summary = json.loads((evaluation / "direct.summary.json").read_text())
                self.assertEqual(summary["runs"], 2)
                self.assertEqual(set(summary["by_run"]), {"0", "1"})
                self.assertEqual(summary["overall"]["spans"]["f1"], 1.0)
                saved = json.loads((evaluation / "tiny.direct.evaluation.json").read_text())
                self.assertEqual(saved["items"][0]["generation"]["raw_output"], raw)


if __name__ == "__main__":
    unittest.main()
