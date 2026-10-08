import unittest

import numpy as np

from scripts.calibrate_patchcore import choose_low_fpr_threshold, metrics


class CalibrationTests(unittest.TestCase):
    def test_fpr_budget_and_recall_tradeoff(self):
        normals = np.arange(1, 21, dtype=float)
        defects = np.array([10., 15., 19., 19.5, 21., 22.])
        labels = [0] * len(normals) + [1] * len(defects)
        scores = np.r_[normals, defects]
        threshold = choose_low_fpr_threshold(labels, scores, .05)
        outcome = metrics(labels, scores, threshold)
        self.assertEqual(outcome['false_positives'], 1)
        self.assertEqual(outcome['recall'], .5)
        lower = np.nextafter(threshold, 0)
        self.assertGreater(metrics(labels, scores, lower)['fpr'], .05)

    def test_tied_normals_strict_comparison_and_determinism(self):
        labels = [0, 0, 0, 0, 1, 1]
        scores = [1., 2., 3., 3., 3., 4.]
        threshold = choose_low_fpr_threshold(labels, scores, .25)
        self.assertEqual(threshold, 3.)
        self.assertEqual(metrics(labels, scores, threshold)['false_positives'], 0)
        self.assertEqual(metrics(labels, scores, threshold)['recall'], .5)
        self.assertEqual(choose_low_fpr_threshold(labels[::-1], scores[::-1], .25), threshold)

    def test_zero_distances_and_extreme_valid_budgets(self):
        labels, scores = [0, 0, 1, 1], [0., 1., 0., 2.]
        self.assertEqual(metrics(labels, scores, choose_low_fpr_threshold(labels, scores, 0))['fpr'], 0)
        threshold = choose_low_fpr_threshold(labels, scores, 1)
        self.assertGreater(threshold, 0)
        self.assertEqual(metrics(labels, scores, threshold)['recall'], .5)

    def test_defect_scores_cannot_shift_normal_budget_threshold(self):
        labels = [0] * 20 + [1] * 3
        normal_scores = list(range(1, 21))
        low_defects = normal_scores + [0., .1, .2]
        high_defects = normal_scores + [100., 200., 300.]
        self.assertEqual(choose_low_fpr_threshold(labels, low_defects),
                         choose_low_fpr_threshold(labels, high_defects))

    def test_invalid_inputs_rejected(self):
        for labels, scores, budget in [([1, 1], [1, 2], .05), ([0, 0], [1, 2], .05),
                                      ([0, 1], [1], .05), ([0, 1], [1, float('nan')], .05),
                                      ([0, 1], [1, float('inf')], .05), ([0, 1], [1, -1], .05),
                                      ([0, 2], [1, 2], .05), ([0, 1], [1, 2], -.1),
                                      ([0, 1], [1, 2], 1.1), ([0, 1], [1, 2], float('nan'))]:
            with self.subTest(labels=labels, scores=scores, budget=budget):
                with self.assertRaises(ValueError):
                    choose_low_fpr_threshold(labels, scores, budget)


if __name__ == '__main__':
    unittest.main()
