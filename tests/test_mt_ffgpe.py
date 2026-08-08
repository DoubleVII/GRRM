import unittest

from inference.prompts import build_ffgpe_prompt, validate_ffgpe_prompt_type
from inference.run_mt_ffgpe import _parse_ffgpe_response


class TestFFGPE(unittest.TestCase):
    def test_prompt_types(self):
        for kind, count in (("fixed_4", 4), ("adaptive", 8), ("fixed_16", 16)):
            text = build_ffgpe_prompt("en", "zh", "hello", count, kind)
            self.assertIn("hello", text)
            validate_ffgpe_prompt_type(kind, count)

    def test_dual_envelope(self):
        output = (
            "<thinking>c</thinking><response>"
            '{"translations":["a","b","c","d"]}'
            "</response><thinking>p</thinking><response>```\nfinal\n```</response>"
        )
        parsed = _parse_ffgpe_response(output, 4, "fixed_4")
        self.assertEqual(parsed["candidates"], ["a", "b", "c", "d"])
        self.assertEqual(parsed["parsed"], "final")

    def test_invalid_envelope(self):
        self.assertIsNone(_parse_ffgpe_response("bad", 4, "fixed_4"))
        output = (
            "<thinking>c</thinking><response>"
            '{"translations":["a","a","c","d"]}'
            "</response><thinking>p</thinking><response>```x```</response>"
        )
        self.assertIsNone(_parse_ffgpe_response(output, 4, "fixed_4"))


if __name__ == "__main__":
    unittest.main()
