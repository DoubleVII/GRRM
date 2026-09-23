import copy
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from inference import run_inst_diverse_mt as mt


ANALYSIS = {
    "source_analysis": "A greeting.",
    "segments": [{"segment_id": 1, "source_span": "Hello", "analysis": "Greeting",
                  "candidates": [{"translation": "你好", "angle": "Neutral greeting"}]}],
}


class FakeTokenizer:
    eos_token = "<eos>"

    def __init__(self):
        self.thinking = []

    def apply_chat_template(self, messages, **kwargs):
        self.thinking.append(kwargs["enable_thinking"])
        return messages[0]["content"]


class FakeModel:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = []

    def generate(self, prompts, params):
        self.calls.append((prompts, params))
        texts = next(self.batches)
        assert len(prompts) == len(texts)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1, 2, 3])])
                for text in texts]


class DiverseTest(unittest.TestCase):
    def test_json_validation(self):
        self.assertEqual(mt.extract_divergent_response(json.dumps(ANALYSIS)), ANALYSIS)
        for text in ("[]", "{}", "```json\n" + json.dumps(ANALYSIS) + "\n```",
                     json.dumps(ANALYSIS) + " extra"):
            self.assertIsNone(mt.extract_divergent_response(text))
        for field, value in (("segment_id", 2), ("source_span", ""), ("candidates", [])):
            bad = copy.deepcopy(ANALYSIS)
            bad["segments"][0][field] = value
            self.assertIsNone(mt.extract_divergent_response(json.dumps(bad)))
        with_confidence = copy.deepcopy(ANALYSIS)
        with_confidence["segments"][0]["candidates"][0]["confidence"] = "high"
        text = json.dumps(with_confidence)
        self.assertIsNone(mt.extract_divergent_response(text))
        self.assertIsNotNone(mt.extract_divergent_response(text, candidate_confidence=True))

    def test_final_requires_complete_wrapper(self):
        self.assertEqual(mt.extract_final_translation("<final_translation>你好</final_translation>"), "你好")
        for text in ("你好", "<final_translation>你好", "<final_translation> </final_translation>",
                     "analysis <final_translation>你好</final_translation>"):
            self.assertIsNone(mt.extract_final_translation(text))

    def test_pipeline_retry_thinking_and_alignment(self):
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                response = json.dumps(ANALYSIS)
                prefix = "<think>reasoning</think>" if thinking else ""
                fake = FakeModel([[prefix + response + "<eos>", "bad"], ["bad again"],
                                  [prefix + "<final_translation>你好</final_translation><eos>"]])
                tokenizer = FakeTokenizer()
                with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SimpleNamespace)}):
                    output = mt.run_pipeline(["Hello", "Hello"], "en", "zh",
                        model=mt.InstructEngine(fake, tokenizer), enable_thinking=thinking, retry=1)
                self.assertEqual([len(call[0]) for call in fake.calls], [2, 1, 1])
                self.assertEqual(output["convergent"]["translations"], ["你好", None])
                self.assertEqual(output["divergent"]["thinking"][0], "reasoning" if thinking else None)
                self.assertEqual(output["divergent"]["output_tokens"], [3, 3])
                self.assertEqual(tokenizer.thinking, [thinking] * 3)
                self.assertIn("mandatory whole-text polish", fake.calls[-1][0][0])
                self.assertNotIn('"confidence"', fake.calls[0][0][0])
                self.assertTrue(fake.calls[0][1].skip_special_tokens is False)
                json.dumps(output)

    def test_eval_runs_metrics_and_kwargs(self):
        import pandas as pd
        from eval import run_inst_diverse_mt_eval as evaluation

        frame = pd.DataFrame(dict(src_text=["Hello", "Bye"], trg_text=["你好", "再见"],
                                  src_lang=["en"] * 2, trg_lang=["zh"] * 2))
        engine = SimpleNamespace(model=object())
        with patch.multiple(evaluation,
            _load_datasets=unittest.mock.DEFAULT, init_inst_model=unittest.mock.DEFAULT,
            run_pipeline=unittest.mock.DEFAULT, _release_vllm_model=unittest.mock.DEFAULT,
            run_bleurt_eval=unittest.mock.DEFAULT, run_oss_eval=unittest.mock.DEFAULT,
            save_results_to_json=unittest.mock.DEFAULT,
            log_results_to_wandb=unittest.mock.DEFAULT) as mocks, patch.object(
                evaluation.run_oss_SQM, "init_oss_model", return_value=object()):
            mocks["_load_datasets"].return_value = (frame, {"test": (0, 2)}, {"test": "en-zh"}, {"test": frame})
            mocks["init_inst_model"].return_value = engine
            mocks["run_pipeline"].return_value = {"convergent": {"translations": ["你好", None, "您好", "再见"]}}
            mocks["run_bleurt_eval"].return_value = [0.8, 0.4, 0.6, 0.8]
            mocks["run_oss_eval"].return_value = [80, 40, 60, 80]
            evaluation.main("test", "model", "name", runs=2, save_results=True,
                            mt_vllm_kwargs={"gpu_memory_utilization": 0.7})
            mocks["init_inst_model"].assert_called_once_with("model", gpu_memory_utilization=0.7)
            self.assertEqual(mocks["run_pipeline"].call_args.args[0], ["Hello", "Bye"] * 2)
            self.assertFalse(mocks["run_pipeline"].call_args.kwargs["candidate_confidence"])
            self.assertEqual(mocks["save_results_to_json"].call_args.kwargs["mt_list_for_runs_nested"],
                             [["你好", "Translation Failed."], ["您好", "再见"]])
            metrics = mocks["log_results_to_wandb"].call_args.args[0]["test"]
            self.assertAlmostEqual(metrics["bleurt"], 0.65)
            self.assertAlmostEqual(metrics["oss"], 65)


if __name__ == "__main__":
    unittest.main()
