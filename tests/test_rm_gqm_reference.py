import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

from data.run_rm_GQM import process_scores
from inference.prompts import GQM_prompt_template, candidate_prompt, get_GQM_prompt, get_task_prompt
from inference.run_rm_GQM import func_call


class ReferenceGQMTest(unittest.TestCase):
    def test_source_only_retains_training_prompt(self):
        expected = GQM_prompt_template.format(
            source_lang="English", target_lang="Chinese", source_text="source",
            task_prompt=get_task_prompt("ranking_score", False),
            candidate_prompts=candidate_prompt.format("A", "one") + candidate_prompt.format("B", "two"),
            reference_prompt="", notes_prompt="",
        )
        self.assertEqual(get_GQM_prompt("en", "zh", "source", ["one", "two"], "ranking_score"), expected)

    def test_inference_uses_evidence_and_retries_same_prompt(self):
        captured = []

        def generate(prompts, params):
            captured.append(prompts)
            response = "invalid" if len(captured) == 1 else "B > A\nB: 9, A: 7"
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=response)])]

        class SamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        tokenizer = types.SimpleNamespace(apply_chat_template=lambda messages, **kwargs: messages[0]["content"])
        with patch.dict(sys.modules, {"vllm": types.SimpleNamespace(SamplingParams=SamplingParams)}):
            result = func_call(
                "unused", ["source"], [["one", "two"]], "en", "zh",
                ref_list=["reference"], comment_list=["expert evidence"],
                model=types.SimpleNamespace(generate=generate), tokenizer=tokenizer, retry=1,
            )
        self.assertEqual(result["scores"], [[7, 9]])
        self.assertEqual(captured[0], captured[1])
        prompt = captured[0][0]
        self.assertIn("Chinese reference:\n```\nreference", prompt)
        self.assertIn("Expert comment:\n```\nexpert evidence", prompt)
        self.assertIn("Judge the candidates using the reference translation and expert comments", prompt)
        self.assertNotIn("Notes:", prompt)

    def test_comment_without_reference_and_empty_evidence(self):
        args = ("en", "zh", "source", ["one", "two"], "ranking_score")
        self.assertIn("Expert comment:", get_GQM_prompt(*args, comment="evidence"))
        self.assertEqual(get_GQM_prompt(*args), get_GQM_prompt(*args, ref_text=" ", comment=" "))

    def test_evidence_length_and_gqmpe_rejected_before_model_loading(self):
        args = ("unused", ["source"], [["one", "two"]], "en", "zh")
        for argument in ("ref_list", "comment_list"):
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(ValueError, argument):
                    func_call(*args, **{argument: []})
        with self.assertRaisesRegex(ValueError, "reference/comment"):
            func_call(*args, task_type="gqmpe", ref_list=["reference"])
        with self.assertRaises(TypeError):
            func_call(*args, notes_list=["old note"])

    def test_data_entry_resumes_and_normalizes_missing_evidence(self):
        frame = pd.DataFrame({
            "src_text": ["done", "pending", "pending2"], "mt_texts": [["one", "two"]] * 3,
            "src_lang": ["en"] * 3, "trg_lang": ["zh"] * 3,
            "ref_text": ["unused", pd.NA, "reference"],
            "comment": ["unused", "evidence", float("nan")],
            "rm_score": [[9, 8], None, None],
        })
        with patch("data.run_rm_GQM.run_rm_GQM.func_call", return_value={
            "scores": [[7, 9], [8, 8]], "responses": ["result", "result2"],
        }) as call:
            result = process_scores(frame, "unused", ref_key="ref_text", comment_key="comment")
        self.assertEqual(call.call_args.kwargs["src_list"], ["pending", "pending2"])
        self.assertEqual(call.call_args.kwargs["ref_list"], [None, "reference"])
        self.assertEqual(call.call_args.kwargs["comment_list"], ["evidence", None])
        self.assertEqual(result["rm_score"].tolist(), [[9, 8], [7, 9], [8, 8]])


if __name__ == "__main__":
    unittest.main()
