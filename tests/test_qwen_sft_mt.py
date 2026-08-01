import json
import unittest

from inference.run_qwen_sft_mt import (
    SftMtEngine,
    generate_validated,
    run_flash_gpe_pipeline,
    run_gpe_pipeline,
    run_scd_pipeline,
)
from inference.sft_mt_protocol import format_sft_output


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        return repr(messages)


class _Candidate:
    def __init__(self, text):
        self.text = text


class _RequestOutput:
    def __init__(self, values):
        self.outputs = [_Candidate(value) for value in values]


class _Model:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
        self.params = []

    def generate(self, prompts, params):
        self.calls.append(prompts)
        self.params.append(params)
        values = self.batches.pop(0)
        self.last_prompts = prompts
        return [_RequestOutput(item) for item in values]


class QwenSftMtTest(unittest.TestCase):
    def test_flash_gpe_generates_four_candidates_in_two_calls(self):
        candidates = ["你好", "您好", "嗨", "你好呀"]
        stage1_raw = format_sft_output(
            "Create diverse translations.",
            json.dumps({"translations": candidates}, ensure_ascii=False),
        )
        stage2_raw = format_sft_output(
            "Select the best rendering.",
            "```translation\n您好\n```",
        )
        model = _Model([[[stage1_raw]], [[stage2_raw]]])
        engine = SftMtEngine(model=model, tokenizer=_Tokenizer())

        result = run_flash_gpe_pipeline(
            ["Hello"],
            ["en"],
            ["zh"],
            engine=engine,
            prompt_type="fixed_4",
            max_candidates=4,
            retry=0,
        )

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result["candidates"], [candidates])
        self.assertEqual(result["usable_candidate_counts"], [4])
        self.assertEqual(result["post_edit"][0][0]["parsed"], "您好")
        prompts = [
            result["candidate_generation"]["messages"][0][0]["content"],
            result["post_edit_messages"][0][0]["content"],
        ]
        for prompt in prompts:
            self.assertNotIn("JSON", prompt)
            self.assertNotIn("schema", prompt)
            self.assertNotIn("<thinking>", prompt)
            self.assertNotIn("<response>", prompt)

    def test_original_gpe_samples_four_independent_completions(self):
        candidate_raw = [
            format_sft_output(
                f"Reason {index}.",
                f"<final_translation>candidate-{index}</final_translation>",
            )
            for index in range(4)
        ]
        final_raw = format_sft_output(
            "Choose the best.", "```translation\ncandidate-1\n```"
        )
        model = _Model([[candidate_raw], [[final_raw]]])
        engine = SftMtEngine(model=model, tokenizer=_Tokenizer())
        result = run_gpe_pipeline(
            ["Hello"], ["en"], ["zh"], engine=engine, sampling_n=4, retry=0
        )
        self.assertEqual(model.params[0].n, 4)
        self.assertEqual(model.params[1].n, 1)
        self.assertEqual(result["usable_candidate_counts"], [4])
        self.assertEqual(result["post_edit"][0][0]["parsed"], "candidate-1")

    def test_invalid_outer_protocol_retries(self):
        model = _Model([
            [["invalid"]],
            [[format_sft_output("reason", "answer")]],
        ])
        engine = SftMtEngine(model=model, tokenizer=_Tokenizer())
        outputs = generate_validated(
            engine,
            [[{"role": "user", "content": "prompt"}]],
            lambda value: value if value == "answer" else None,
            temperature=0.3,
            top_p=0.8,
            max_tokens=100,
            retry=1,
        )
        self.assertEqual(outputs[0][0]["parsed"], "answer")
        self.assertEqual(outputs[0][0]["thinking"], "reason")

    def test_scd_stage2_contains_stage1_assistant_history(self):
        stage1_response = """{
          "source_analysis": "greeting",
          "segments": [{
            "segment_id": 1,
            "source_span": "Hello",
            "analysis": "greeting",
            "candidates": [{"translation": "你好", "angle": "neutral"}]
          }]
        }"""
        stage1_raw = format_sft_output("analyze", stage1_response)
        stage2_raw = format_sft_output(
            "compose", "<final_translation>你好</final_translation>"
        )
        model = _Model([[[stage1_raw]], [[stage2_raw]]])
        engine = SftMtEngine(model=model, tokenizer=_Tokenizer())
        result = run_scd_pipeline(
            ["Hello"], ["en"], ["zh"], engine=engine, retry=0
        )
        messages = result["stage2_messages"][0]
        self.assertEqual([item["role"] for item in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], stage1_raw)
        self.assertNotIn(stage1_response, messages[2]["content"])
        self.assertEqual(result["stage2"][0][0]["parsed"], "你好")


if __name__ == "__main__":
    unittest.main()
