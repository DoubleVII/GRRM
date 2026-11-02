import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import pandas.core.arrays.arrow.extension_types
import openai_harmony

from data.run_oss_hard_annotation import load_items, main
from inference.run_oss_hard_annotation import annotate, validate_annotation


def item():
    return dict(src_text="我的老板除了CPU我什么也不会。", src_lang="zh", trg_lang="en")


def annotation():
    return dict(analysis="Workplace slang", score=8,
                spans=[dict(source_span="CPU我", score=8, explanation="Gaslighting, not hardware")])


class OssHardAnnotationTests(unittest.TestCase):
    def test_validator_rejects_malformed_scores_quotes_duplicates_and_long_spans(self):
        good = annotation()
        self.assertEqual(validate_annotation(good, item()), good)
        self.assertIsNotNone(validate_annotation(dict(analysis="Routine", score=0, spans=[]), item()))
        invalid = [dict(good, score=True), dict(good, score=7), dict(good, score=11),
                   dict(good, spans=[]), dict(good, spans=good["spans"] * 2),
                   dict(good, analysis="")]
        for update in [dict(source_span="cpu我"), dict(source_span="不存在"),
                       dict(score=False), dict(score=0), dict(explanation="")]:
            invalid.append(dict(good, spans=[dict(good["spans"][0], **update)]))
        for value in invalid:
            self.assertIsNone(validate_annotation(value, item()))
        self.assertIsNone(validate_annotation(good, item(), max_span_chars=3))
        spaced = dict(src_text="one two three")
        value = dict(analysis="x", score=2, spans=[dict(source_span=spaced["src_text"], score=2, explanation="x")])
        self.assertIsNone(validate_annotation(value, spaced, max_span_words=2))

    def test_raw_dataset_cli_artifacts_and_preflight(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # Deliberately no reference/candidate columns; extra notes must not leak.
            pd.DataFrame([dict(item(), comment="private note")]).to_parquet(root / "tiny.parquet")
            def fake_annotate(items, *args, **kwargs):
                self.assertEqual(set(items[0]), {"data_id", "row_id", "src_text", "src_lang", "trg_lang"})
                return [dict(value, status="ok", result=annotation(), generation={}, error=None) for value in items]
            with patch("data.run_oss_hard_annotation.annotate", side_effect=fake_annotate) as generate:
                main("tiny", root / "out", data_dir=root)
                payload = json.loads((root / "out/tiny.oss.difficulty_annotations.json").read_text())
                self.assertEqual(payload["config"]["model_path"], "openai/gpt-oss-120b")
                self.assertEqual(payload["summary"], dict(total=1, valid=1, failed=0))
                self.assertEqual(payload["stage"], "difficulty_annotations")
                with self.assertRaises(FileExistsError):
                    main("new,tiny", root / "out", data_dir=root)
                self.assertEqual(generate.call_count, 1)
            with self.assertRaises(ValueError):
                load_items("tiny", root, max_samples=-1)

    def test_oss_retry_and_candidate_independence(self):
        class FakeModel:
            def __init__(self):
                self.calls = []
            def generate(self, prompts, sampling_params):
                self.calls.append(prompts)
                # First request fails score consistency, then succeeds.
                value = dict(annotation(), score=2) if len(self.calls) == 1 else annotation()
                text = json.dumps(value)
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=text)]) for _ in prompts]
        model = FakeModel()
        encoding = SimpleNamespace(stop_tokens_for_assistant_actions=lambda: [])
        fake_vllm = SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict("sys.modules", {"vllm": fake_vllm}), \
                patch("inference.run_oss_SQM.load_encoding", return_value=encoding), \
                patch("inference.run_oss_diverse_mt._prepare_inputs", side_effect=lambda prompts, *args: prompts), \
                patch("inference.run_oss_diverse_mt._response_text", side_effect=lambda tokens, enc: (None, tokens)):
            output = annotate([dict(item(), candidate="candidate sentinel", ref_text="reference sentinel",
                                    comment="comment sentinel")], model=model, retry=1)
            with self.assertWarns(UserWarning):
                failed = annotate([item()], model=FakeModel(), retry=0)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(output[0]["status"], "ok")
        self.assertEqual(output[0]["result"], annotation())
        self.assertEqual(failed[0]["status"], "failed")
        self.assertIsNone(failed[0]["result"])
        self.assertIsNotNone(failed[0]["generation"]["response"])
        for field in ("candidate", "ref_text", "comment"):
            self.assertNotIn(field, output[0])
        for text in model.calls[0]:
            self.assertNotIn("sentinel", text)
            self.assertIn("48 characters and 8 whitespace-separated words", text)

    def test_relocated_configured_data_and_language_defaults(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "ltb-v1-eval-scarcity"
            folder.mkdir()
            pd.DataFrame([dict(item(), ltb_id=111)]).to_parquet(folder / "high_resource.parquet")
            rows = load_items("ltb_v1_eval_scarcity_high", root)
            self.assertEqual(rows[0]["ltb_id"], 111)
            pd.DataFrame([dict(src_text="A plain headline")]).to_parquet(root / "wmt24_enzh.parquet")
            rows = load_items("wmt24_enzh", root)
            self.assertEqual((rows[0]["src_lang"], rows[0]["trg_lang"]), ("en", "zh"))


if __name__ == "__main__":
    unittest.main()
