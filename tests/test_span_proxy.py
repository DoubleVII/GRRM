import itertools
import random
import unittest
from unittest.mock import patch

from eval.span_proxy import SpanSource, maximum_weight_matching, score_spans as _score_spans


def score_spans(*args, **options):
    return _score_spans(*args, **{"unit": "token", "method": "containment", **options})


class SpanProxyTests(unittest.TestCase):
    def test_selected_default_is_asymmetric(self):
        source = 'one two three four'
        self.assertAlmostEqual(_score_spans(source, [source], ['one'])['f1'], (3/15)**0.5)
        self.assertAlmostEqual(_score_spans(source, ['one'], [source])['f1'], (3/15)**0.25)

    def test_evaluation_uses_no_judge_and_summarizes_soft_counts(self):
        from eval.difficulty_eval_core import evaluate
        def record(row, quote):
            return dict(data_id='tiny', row_id=row, src_text='one two three four', src_lang='en',
                        trg_lang='zh', status='ok', result=dict(analysis='difficulty', score=5 if quote else 0,
                            spans=[dict(source_span=quote, score=5, explanation='difficulty')] if quote else []))
        predictions = [record(0, 'one two three four'), record(1, '')]
        references = [record(0, 'one'), record(1, '')]
        with patch('eval.difficulty_eval_core.generate', side_effect=AssertionError('LLM was called')):
            items, summary = evaluate(predictions, references, span_backend='heuristic')
        expected = (3/15)**0.5
        self.assertAlmostEqual(items[0]['metrics']['f1'], expected)
        self.assertIsNone(items[0]['generation'])
        self.assertEqual(items[0]['result']['backend'], 'heuristic')
        self.assertEqual(items[1]['metrics']['f1'], 1)
        self.assertAlmostEqual(summary['overall']['spans']['f1'], expected)
        self.assertEqual(summary['overall']['scores']['mae'], 0)

    def test_empty_and_ungrounded(self):
        self.assertEqual(score_spans('hello world', [], [])['f1'], 1)
        self.assertEqual(score_spans('hello world', ['hello'], [])['f1'], 0)
        result = score_spans('hello world', ['invented'], ['world'])
        self.assertEqual(result['f1'], 0)
        self.assertEqual(result['ungrounded_prediction'], 1)
        self.assertAlmostEqual(score_spans('the cat', ['he'], ['the'])['f1'], (2/3)**0.5)

    def test_positions_and_normalization(self):
        # Character bags would overlap, but these are different source locations.
        self.assertEqual(score_spans('abc bca', ['abc'], ['bca'])['f1'], 0)
        self.assertEqual(score_spans('ＡＢＣ — 中国', ['abc中国'], ['ＡＢＣ — 中国'])['f1'], 1)
        self.assertEqual(score_spans('hello, world!', ['HELLO WORLD'], ['hello, world'])['f1'], 1)
        self.assertEqual(score_spans('折扣与折旧', ['折扣'], ['折旧'])['f1'], 0)
        self.assertAlmostEqual(score_spans('折扣与折旧', ['折扣与'], ['与折旧'])['f1'], 1/3)
        self.assertEqual(score_spans('お金壊してきてちょうだい', ['壊してきて'], ['壊してきて'])['f1'], 1)
        self.assertEqual(score_spans('Rhabarberbarbarabarbarbarenbartbarbierbier', ['barbarenbart'], ['barbarenbart'])['f1'], 1)
        self.assertEqual(score_spans('cat and cat dog', ['cat'], ['cat dog'])['f1'], 2 ** -0.5)

    def test_length_and_duplicate_penalties(self):
        source = 'one two three four five six seven eight nine ten'
        compact = score_spans(source, ['one'], ['one'])['f1']
        long = score_spans(source, [source], ['one'])['f1']
        self.assertEqual(compact, 1)
        self.assertAlmostEqual(long, 10 ** -0.5)
        self.assertAlmostEqual(score_spans(source, ['one', 'one'], ['one'])['f1'], 2/3)
        self.assertAlmostEqual(score_spans(source, ['one two'], ['one', 'two'])['f1'], 2 * 2 ** -0.5 / 3)
        scores = [score_spans(source, [' '.join(source.split()[:k])], ['one'])['f1'] for k in range(1, 11)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_units_and_methods(self):
        source = 'lengthy a'
        self.assertAlmostEqual(score_spans(source, [source], ['a'], method='dice')['f1'], 2/3)
        self.assertAlmostEqual(score_spans(source, [source], ['a'], unit='char', method='dice')['f1'], 2/9)
        self.assertEqual(score_spans(source, [source], ['a'], method='exact')['f1'], 0)
        context = SpanSource(source)
        self.assertEqual(score_spans(context, ['a'], ['a'])['f1'], 1)
        for options in (dict(unit='bad'), dict(method='bad'), dict(gamma=0), dict(gamma=float('nan')), dict(threshold=0)):
            with self.assertRaises(ValueError):
                score_spans(source, [], [], **options)

    def test_assignment_matches_exhaustive_search(self):
        rng = random.Random(1337)
        self.assertEqual(maximum_weight_matching([]), [])
        self.assertEqual(maximum_weight_matching([[]]), [])
        for n in range(1, 5):
            for m in range(1, 5):
                for _ in range(10):
                    weights = [[rng.random() if rng.random() > .2 else 0 for _ in range(m)] for _ in range(n)]
                    actual = sum(w for _, _, w in maximum_weight_matching(weights))
                    matrix = weights if n <= m else list(zip(*weights))
                    expected = max(sum(matrix[i][j] for i, j in enumerate(assignment))
                                   for assignment in itertools.permutations(range(len(matrix[0])), len(matrix)))
                    self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
