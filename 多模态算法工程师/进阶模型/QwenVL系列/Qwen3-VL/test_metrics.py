import unittest

from kie_metrics import field_counts, micro_f1, parse_json_output
from vqa_metrics import anls_score, chartqa_relaxed_accuracy


class VQAMetricTests(unittest.TestCase):
    def test_anls_threshold_and_multiple_references(self):
        self.assertEqual(anls_score("hello", ["unrelated", "hello"]), 1.0)
        self.assertEqual(anls_score("a", ["abcdefgh"]), 0.0)

    def test_chartqa_numeric_tolerance_and_percent(self):
        self.assertEqual(chartqa_relaxed_accuracy("104", "100"), 1.0)
        self.assertEqual(chartqa_relaxed_accuracy("106", "100"), 0.0)
        self.assertEqual(chartqa_relaxed_accuracy("50%", "0.5"), 1.0)
        self.assertEqual(chartqa_relaxed_accuracy("$1,000.", "1000"), 1.0)

    def test_kie_micro_f1(self):
        reference = {"invoice": "A-1", "total": "20.00"}
        prediction = parse_json_output('```json\n{"invoice":"A-1","total":"21.00"}\n```')
        result = micro_f1([field_counts(prediction, reference)])
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
