import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from inference.prompts import (
    OSS_GQM_PROMPT_TEMPLATE,
    candidate_prompt,
    get_oss_GQM_prompt,
)
from inference.run_oss_GQM import func_call, prepare_vllm_inputs


class OssGQMReferenceTest(unittest.TestCase):
    def test_prompt_without_evidence_is_unchanged(self):
        expected = OSS_GQM_PROMPT_TEMPLATE.format(
            source_lang="English", target_lang="Chinese", source_text="source",
            analysis_instruction="Output your analysis and comparison before the final JSON object.",
            candidate_prompts=candidate_prompt.format("A", "one") + candidate_prompt.format("B", "two"),
        )
        self.assertEqual(get_oss_GQM_prompt("en", "zh", "source", ["one", "two"]), expected)

    def test_prompt_supports_each_evidence_type_and_json_only(self):
        args = ("en", "zh", "source", ["one", "two"])
        for ref, comment in (("reference", None), (None, "evidence"), ("reference", "evidence")):
            with self.subTest(ref=ref, comment=comment):
                prompt = get_oss_GQM_prompt(*args, explicit_analysis=False, ref_text=ref, comment=comment)
                self.assertIn("Output only the final JSON object and no analysis.", prompt)
                self.assertEqual("Chinese reference:" in prompt, ref is not None)
                self.assertEqual("Expert comment:" in prompt, comment is not None)
                self.assertIn("Judge the candidates using the reference translation and expert comments", prompt)
        self.assertEqual(get_oss_GQM_prompt(*args), get_oss_GQM_prompt(*args, ref_text=" ", comment=" "))

    def test_inference_forwards_evidence_and_retries_prepared_input(self):
        class SamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        result = {"scores": [7, 9], "analysis": "comparison", "thinking": "thinking", "response": '{"A":7,"B":9}'}
        with (
            patch.dict(sys.modules, {"vllm": types.SimpleNamespace(SamplingParams=SamplingParams)}),
            patch("inference.run_oss_GQM.load_encoding", return_value=MagicMock()),
            patch("inference.run_oss_GQM.get_oss_GQM_prompt", wraps=get_oss_GQM_prompt) as build,
            patch("inference.run_oss_GQM.run_generate", side_effect=[[None], [result]]) as generate,
        ):
            output = func_call(
                ["source"], [["one", "two"]], "en", "zh", model=object(), retry=1,
                ref_list=["reference"], comment_list=["evidence"],
            )
        self.assertEqual(output["scores"], [[7, 9]])
        self.assertEqual(build.call_args.kwargs["ref_text"], "reference")
        self.assertEqual(build.call_args.kwargs["comment"], "evidence")
        self.assertEqual(generate.call_args_list[0].args[1], generate.call_args_list[1].args[1])

    def test_lengths_are_validated_before_inference(self):
        for argument in ("ref_list", "comment_list"):
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(ValueError, argument):
                    func_call(["source"], [["one", "two"]], "en", "zh", **{argument: []})
                with self.assertRaisesRegex(ValueError, argument):
                    prepare_vllm_inputs(["source"], [["one", "two"]], ["en"], ["zh"], MagicMock(), **{argument: []})


if __name__ == "__main__":
    unittest.main()
