import unittest

import pandas as pd

from utils.helpers import build_notes_list


class BuildNotesListTest(unittest.TestCase):
    def test_normalizes_notes_and_repeats_in_run_major_order(self):
        frame = pd.DataFrame({
            "notes": [" useful note ", "   ", None, 12],
            "difficulty": [0, 999, 5, 1],
        })

        self.assertEqual(
            build_notes_list(frame, runs=2),
            [" useful note ", None, None, None] * 2,
        )

    def test_missing_notes_column_returns_none_for_each_item(self):
        frame = pd.DataFrame({"src_text": ["one", "two"]})

        self.assertEqual(build_notes_list(frame, runs=3), [None] * 6)


if __name__ == "__main__":
    unittest.main()
