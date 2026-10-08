import unittest
import numpy as np
from scripts.train_mpdd import normal_threshold, validate_manifest, test_metrics


def manifest():
    return [{'path': f'{i}.png', 'category': 'connector', 'source_split': 'test' if split == 'test' else 'train',
             'split': split, 'label': 'normal', 'sha256': str(i)} for i, split in enumerate(('train', 'validation', 'test'))]


class MPDDTests(unittest.TestCase):
    def test_small_normal_budget_zero_alarm_with_strict_ties(self):
        scores = [1., 2., 3., 3., 2.]
        threshold = normal_threshold(scores)
        self.assertEqual(threshold, 3.)
        self.assertEqual(sum(value > threshold for value in scores), 0)
        self.assertGreater(sum(value > np.nextafter(threshold, 0) for value in scores), 0)

    def test_normal_budget_without_defect_validation(self):
        scores = np.arange(1, 101)
        threshold = normal_threshold(scores)
        self.assertEqual(sum(scores > threshold), 5)
        self.assertGreater(sum(scores > np.nextafter(threshold, 0)), 5)
        for invalid in [[], [float('nan')], [-1], [[1, 2]]]:
            with self.assertRaises(ValueError):
                normal_threshold(invalid)

    def test_manifest_rejects_label_leakage_and_crosssplit_duplicates(self):
        validate_manifest(manifest())
        for alter in ('defect_train', 'same_hash', 'official_test_train'):
            records = manifest()
            if alter == 'defect_train': records[0]['label'] = 'defect'
            elif alter == 'same_hash': records[1]['sha256'] = records[0]['sha256']
            else: records[0]['source_split'] = 'test'
            with self.assertRaises(ValueError):
                validate_manifest(records)

    def test_metrics_false_alarm_recall_tradeoff(self):
        records = [{'label': label} for label in ['normal', 'normal', 'defect', 'defect']]
        metrics = test_metrics(records, [1., 2., 2., 3.], 2.)
        self.assertEqual(metrics['confusion_matrix'], [[2, 0], [1, 1]])
        self.assertEqual(metrics['normal_fpr'], 0.)
        self.assertEqual(metrics['recall'], .5)


if __name__ == '__main__':
    unittest.main()
