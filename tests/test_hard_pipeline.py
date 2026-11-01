import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import pandas.core.arrays.arrow.extension_types  # initialize before patch.dict restores sys.modules

from data.run_inst_hard_candidates import load_items, main as candidate_main
from data.run_hard_golden import main as golden_main
from eval.hard_eval_core import (evaluate,
                                 maximum_matching, metrics, summarize, validate_pairs)
from eval.run_hard_eval import main as eval_main
from inference.hard_generation import generate
from inference.hard_prompts import build_prompt, golden_prompt, matching_prompt
from inference.run_inst_hard_prediction import main as prediction_main, predict
from utils.hard_protocol import (align, artifact_path, load, load_dataset_artifacts,
                                 output_paths, parse_data_ids, result_item, save, validate_spans)


def sample(row_id=0):
    return dict(data_id="tiny", row_id=row_id, src_text="今日餐品七五折", src_lang="zh",
                trg_lang="en", ref_text="25% off today's food", comment="discount guidance",
                candidate="75% off today's food")


def spans(golden=False):
    span = dict(source_span="七五折", explanation="discount conversion")
    if golden:
        span.update(translation_span="75%", error_type="other")
    return dict(analysis="discount", spans=[span])


def successful(item, value):
    return result_item(item, dict(parsed=value, response=json.dumps(value), thinking=None))


class FakeModel:
    def __init__(self, backend="inst", invalid_first=False):
        self.backend = backend
        self.invalid_first = invalid_first
        self.calls = []

    def generate(self, prompts, sampling_params):
        self.calls.append(prompts)
        outputs = []
        for prompt in prompts:
            if self.invalid_first and len(self.calls) == 1:
                text = "invalid"
            elif "Return ALL compatible edges" in prompt:
                text = json.dumps(dict(analysis="same discount error", pairs=[[0, 0]]))
            elif "Mark actual translation errors" in prompt:
                text = json.dumps(spans(golden=True))
            elif "Return only JSON" in prompt:
                text = json.dumps(spans())
            else:
                text = "# Step-by-step Analysis\n\nDiscount.\n\n# Final Translation\n\n```\n75% off today's food\n```"
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=text)]))
        return outputs


def engine(model):
    return SimpleNamespace(model=model, tokenizer=SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: messages[0]["content"]))


class HardProtocolTests(unittest.TestCase):
    def test_prediction_input_isolation(self):
        item = sample()
        direct, qe = build_prompt(item, "direct"), build_prompt(item, "qe")
        for prompt in (direct, qe):
            self.assertNotIn(item["ref_text"], prompt)
            self.assertNotIn(item["comment"], prompt)
        for prompt in (direct, qe, golden_prompt(item)):
            self.assertNotIn('"start"', prompt)
            self.assertNotIn('"end"', prompt)
            self.assertNotIn("character offsets", prompt)
        self.assertNotIn(item["candidate"], direct)
        self.assertIn(item["candidate"], qe)
        self.assertIn(item["comment"], golden_prompt(item))
        self.assertIn(item["comment"], matching_prompt(successful(item, spans()), successful(item, spans(True))))
        item["comment"] = None
        self.assertNotIn('"comment"', golden_prompt(item))
        with self.assertRaises(ValueError):
            build_prompt(item, "unknown")

    def test_source_quote_validation_and_error_types(self):
        item = sample()
        self.assertIsNotNone(validate_spans(spans(), item))
        self.assertIsNotNone(validate_spans(spans(True), item, True))
        for invalid_quote in ("七折", "", " ", None, 75):
            invalid = spans()
            invalid["spans"][0]["source_span"] = invalid_quote
            self.assertIsNone(validate_spans(invalid, item))
        omission = spans(True)
        omission["spans"][0].update(translation_span="", error_type="omission")
        self.assertIsNotNone(validate_spans(omission, item, True))
        omission["spans"][0]["error_type"] = "addition"
        self.assertIsNone(validate_spans(omission, item, True))
        omission["spans"][0]["translation_span"] = "75%"
        self.assertIsNotNone(validate_spans(omission, item, True))
        omission["spans"][0]["translation_span"] = "not in candidate"
        self.assertIsNone(validate_spans(omission, item, True))
        self.assertIsNotNone(validate_spans(dict(analysis="none", spans=[]), item))
        duplicate = spans()
        duplicate["spans"] *= 2
        self.assertIsNone(validate_spans(duplicate, item))

    def test_artifact_validation_and_alignment(self):
        item = sample()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "candidates.json"
            save(path, "candidates", {}, [item])
            self.assertEqual(load(path, "candidates")["items"], [item])
            with self.assertRaises(FileExistsError):
                save(path, "candidates", {}, [])
            with self.assertRaises(ValueError):
                load(path, "golden")
        self.assertEqual(len(align([item], [item])), 1)
        for changed in ({**item, "candidate": "different"}, {**item, "ref_text": "different"},
                        {**item, "row_id": 1}):
            with self.assertRaises(ValueError):
                align([item], [changed])
        with self.assertRaises(ValueError):
            align([item, item], [item])

    def test_maximum_matching_requires_reassignment(self):
        # Greedy first-edge matching gives one TP; augmenting paths give two.
        result = maximum_matching([[0, 0], [0, 1], [1, 0]], 2)
        self.assertEqual(result, [[0, 1], [1, 0]])
        self.assertEqual(len(maximum_matching([[0, 0], [1, 0]], 2)), 1)
        self.assertIsNone(validate_pairs(dict(analysis="x", pairs=[[0, 2]]), 1, 1))
        self.assertIsNone(validate_pairs(dict(analysis="x", pairs=[[True, 0]]), 1, 1))
        self.assertIsNone(validate_pairs(dict(analysis="x", pairs=[[0, 0], [0, 0]]), 1, 1))

    def test_empty_and_failed_samples(self):
        empty = dict(analysis="none", spans=[])
        prediction = successful(sample(), empty)
        gold = successful(sample(), empty)
        with patch("eval.hard_eval_core.generate", return_value=[]) as generator:
            items, summary = evaluate([prediction], [gold])
        self.assertEqual(items[0]["metrics"]["f1"], 1)
        self.assertEqual(summary["overall"]["f1"], 0)
        self.assertEqual(summary["overall"]["both_empty"], 1)
        self.assertEqual(generator.call_args.args[0], [])
        with patch("eval.hard_eval_core.generate", return_value=[]):
            items, summary = evaluate([successful(sample(), spans())], [gold])
        self.assertEqual(items[0]["metrics"], metrics(0, 1, 0))
        with patch("eval.hard_eval_core.generate", return_value=[]):
            items, summary = evaluate([prediction], [successful(sample(), spans(True))])
        self.assertEqual(items[0]["metrics"], metrics(0, 0, 1))
        failed = result_item(sample(), error="parse failure")
        with patch("eval.hard_eval_core.generate", return_value=[]):
            items, summary = evaluate([failed], [gold])
        self.assertEqual(summary["overall"]["coverage"], 0)
        self.assertIsNone(items[0]["metrics"])
        combined = [dict(status="ok", metrics=metrics(1, 2, 0), both_empty=False),
                    dict(status="ok", metrics=metrics(2, 0, 1), both_empty=False),
                    dict(status="failed", metrics=None)]
        summary = summarize(combined)
        self.assertEqual(summary["precision"], 3 / 5)
        self.assertEqual(summary["recall"], 3 / 4)
        self.assertEqual(summary["coverage"], 2 / 3)

    def test_instruct_retry_and_candidate_failure(self):
        fake_vllm = SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
        model = FakeModel(invalid_first=True)
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            output = predict([sample()], "direct", "test", model=engine(model), retry=1)
        self.assertEqual(output[0]["status"], "ok")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(output[0]["generation"]["raw_output"], json.dumps(spans()))
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            output = predict([sample()], "direct", "test", model=engine(FakeModel(invalid_first=True)), retry=0)
        self.assertEqual(output[0]["status"], "failed")
        self.assertEqual(output[0]["generation"]["raw_output"], "invalid")
        with patch("inference.run_inst_hard_prediction.generate", return_value=[]) as generator:
            output = predict([{**sample(), "candidate": None}], "direct", "test")
        self.assertEqual(output[0]["error"], "Candidate translation failed")
        self.assertEqual(generator.call_args.args[0], [])

    def test_dataset_selection(self):
        with TemporaryDirectory() as directory:
            frame = pd.DataFrame([dict(src_text="今日餐品七五折", src_lang="zh", trg_lang="en",
                                       trg_text="25% off", comment=comment) for comment in (None, "", "note")])
            for did in ("a", "b"):
                frame.to_parquet(Path(directory) / f"{did}.parquet", index=False)
            items = load_items("a,b", directory, 2)
            self.assertEqual([(item["data_id"], item["row_id"]) for item in items],
                             [("a", 0), ("a", 1), ("b", 0), ("b", 1)])
            self.assertTrue(all(item["comment"] is None for item in items))
            with self.assertRaises(ValueError):
                load_items("a,a", directory)

    def test_dataset_artifact_names_and_validation(self):
        self.assertEqual(parse_data_ids("a, b"), ("a", "b"))
        self.assertEqual(parse_data_ids(["a", "b"]), ("a", "b"))
        for invalid in ("", "a,a", "a,", ["../a"], ["/a"], [None]):
            with self.assertRaises(ValueError):
                parse_data_ids(invalid)
        with TemporaryDirectory() as directory:
            path = artifact_path(directory, "tiny", "candidates")
            self.assertEqual(path.name, "tiny.candidates.json")
            save(path, "candidates", {}, [sample()])
            items, _ = load_dataset_artifacts(("tiny",), directory, "candidates")
            self.assertEqual(items, [sample()])
            with self.assertRaises(FileNotFoundError):
                load_dataset_artifacts(("missing",), directory, "candidates")
            wrong_path = artifact_path(directory, "wrong", "candidates")
            save(wrong_path, "candidates", {}, [sample()])
            with self.assertRaises(ValueError):
                load_dataset_artifacts(("wrong",), directory, "candidates")
            prediction_path = artifact_path(directory, "tiny", "predictions", "qe")
            save(prediction_path, "predictions", {"mode": "direct"}, [successful(sample(), spans())])
            with self.assertRaises(ValueError):
                load_dataset_artifacts(("tiny",), directory, "predictions", "qe")
            # Preflight every dataset before any generation or writing.
            with patch("data.run_inst_hard_candidates.make_candidates") as generator:
                with self.assertRaises(FileExistsError):
                    candidate_main("new,tiny", directory, "test", data_dir=directory)
            generator.assert_not_called()
            self.assertFalse(artifact_path(directory, "new", "candidates").exists())

    def test_indexed_validation_and_retry_only_failed_items(self):
        class MixedModel:
            def __init__(self):
                self.calls = []

            def generate(self, prompts, params):
                self.calls.append(prompts)
                values = ([dict(index=0), dict(index=99)] if len(self.calls) == 1
                          else [dict(index=1)])
                return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(value))])
                        for value in values]

        model = MixedModel()
        fake_vllm = SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            output = generate(["first", "second"],
                [lambda value: value if value == dict(index=0) else None,
                 lambda value: value if value == dict(index=1) else None],
                backend="inst", model_path="test", model=engine(model), retry=1)
        self.assertEqual(model.calls, [["first", "second"], ["second"]])
        self.assertEqual([value["parsed"] for value in output], [dict(index=0), dict(index=1)])

    def test_judge_failure_excluded_and_raw_response_preserved(self):
        failure = dict(parsed=None, response="malformed", thinking="analysis")
        with patch("eval.hard_eval_core.generate", return_value=[failure]):
            items, summary = evaluate([successful(sample(), spans())],
                                      [successful(sample(), spans(True))])
        self.assertEqual(summary["overall"]["valid"], 0)
        self.assertIsNone(items[0]["metrics"])
        self.assertEqual(items[0]["generation"]["response"], "malformed")

    def test_all_four_stages_with_fake_models(self):
        fake_vllm = SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
        inst, oss = FakeModel(), FakeModel("oss", invalid_first=True)
        encoding = SimpleNamespace(stop_tokens_for_assistant_actions=lambda: [])
        with TemporaryDirectory() as directory, \
                patch.dict("sys.modules", {"vllm": fake_vllm}), \
                patch("inference.run_inst_mt.init_inst_model", return_value=engine(inst)), \
                patch("inference.run_oss_SQM.init_oss_model", return_value=oss), \
                patch("inference.run_oss_SQM.load_encoding", return_value=encoding), \
                patch("inference.run_oss_diverse_mt._prepare_inputs", side_effect=lambda prompts, *args: prompts), \
                patch("inference.run_oss_diverse_mt._response_text", side_effect=lambda tokens, enc: (None, tokens)):
            root = Path(directory)
            original = sample()
            for did in ("tiny", "tiny2"):
                pd.DataFrame([dict(src_text=original["src_text"], src_lang="zh", trg_lang="en",
                                   trg_text=original["ref_text"], comment=original["comment"])])\
                    .to_parquet(root / f"{did}.parquet", index=False)
            candidates_dir, golden_dir = str(root / "candidates"), str(root / "golden")
            predictions_dir, evaluation_dir = str(root / "predictions"), str(root / "evaluation")
            candidate_main("tiny,tiny2", candidates_dir, "test-inst", data_dir=directory)
            golden_main(["tiny", "tiny2"], candidates_dir, golden_dir, retry=1)
            self.assertEqual(len(oss.calls), 2)  # malformed JSON was retried
            for mode in ("direct", "qe"):
                prediction_main("tiny,tiny2", candidates_dir, predictions_dir, "test-inst", mode=mode)
                with patch("builtins.print") as console:
                    eval_main("tiny,tiny2", predictions_dir, golden_dir, evaluation_dir, mode=mode)
                aggregate = json.loads(console.call_args.args[0])
                self.assertEqual(aggregate["overall"]["tp"], 2)
                self.assertEqual(aggregate["overall"]["total"], 2)
                for did in ("tiny", "tiny2"):
                    payload = load(artifact_path(evaluation_dir, did, "evaluation", mode), "evaluation")
                    self.assertEqual(payload["summary"]["overall"]["f1"], 1)
                    self.assertEqual(payload["summary"]["by_dataset"][did]["tp"], 1)
                    self.assertEqual(len(payload["items"]), 1)
                    self.assertEqual(payload["items"][0]["data_id"], did)
                    self.assertEqual(payload["items"][0]["result"]["matching"], [[0, 0]])
                    self.assertEqual(payload["items"][0]["candidate"], original["candidate"])
            # One translation call, one prediction call for each mode.
            self.assertEqual(len(inst.calls), 3)
            self.assertTrue(all(len(batch) == 2 for batch in inst.calls))


if __name__ == "__main__":
    unittest.main()
