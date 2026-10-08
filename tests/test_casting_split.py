import json
import unittest
from scripts.prepare_casting import assign_splits, MANIFEST
from scripts.train_patchcore import choose_threshold


class CastingSplitTests(unittest.TestCase):
    def test_exact_duplicate_groups_never_cross_splits(self):
        records = [{'path': f'{label}/{i}.jpg', 'sha256': f'{label}-{i//2}', 'label': label}
                   for label in ('normal','defect') for i in range(40)]
        first = assign_splits(records)
        self.assertEqual(first, assign_splits(list(reversed(records))))
        hashes = {}
        for record in first:
            hashes.setdefault(record['sha256'], set()).add(record['split'])
            self.assertFalse(record['split'] == 'train' and record['label'] == 'defect')
        self.assertTrue(all(len(splits) == 1 for splits in hashes.values()))

    def test_conflicting_labels_for_identical_bytes_rejected(self):
        with self.assertRaises(ValueError):
            assign_splits([{'path':'a','sha256':'same','label':'normal'}, {'path':'b','sha256':'same','label':'defect'}])

    def test_actual_manifest_counts_and_hash_disjointness(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest['counts'], {'train':{'normal':363,'defect':0},
                                            'validation':{'normal':78,'defect':390}, 'test':{'normal':78,'defect':391}})
        grouped = {}
        for record in manifest['records']:
            grouped.setdefault(record['sha256'],set()).add(record['split'])
            self.assertTrue(record['path'].startswith('data/raw/'))
            self.assertEqual(len(record['sha256']),64)
        self.assertTrue(all(len(splits) == 1 for splits in grouped.values()))
        self.assertFalse(manifest['physical_part_ids_available'])
        self.assertFalse(manifest['near_duplicate_check_performed'])

    def test_threshold_matches_runtime_strict_comparison(self):
        threshold, selection = choose_threshold([0,0,1,1], [1.,2.,3.,4.])
        self.assertEqual(threshold,2.)
        self.assertEqual(selection['f1'],1.)
        self.assertEqual([score > threshold for score in [1.,2.,3.,4.]], [False,False,True,True])


if __name__ == '__main__':
    unittest.main()
