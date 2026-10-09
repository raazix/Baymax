import glob
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_rust import is_mosaic, seam_scores, targets  # noqa: E402
from core.upload_inspection import _apply_severity, _corrosion_defects  # noqa: E402
from vision import rust_classifier  # noqa: E402

RAW = sorted(glob.glob('data/raw/rust_detection/valid/*.jpg'))
COLOUR = np.dstack([np.full((64, 64), 40, np.uint8), np.full((64, 64), 90, np.uint8), np.full((64, 64), 170, np.uint8)])


class RustPreparationTests(unittest.TestCase):
    def test_targets_from_image_tags(self):
        self.assertEqual(targets({'mild-corrosion': 1, 'severe-corrosion': 1}), (True, 'severe'))
        self.assertEqual(targets({'corroded-part': 1}), (True, 'ungraded'))
        self.assertEqual(targets({'car': 1}), (False, 'none'))       # a car is an object, not corrosion

    @unittest.skipUnless(len(RAW) >= 8, 'rust dataset not extracted')
    def test_synthetic_2x2_mosaic_is_detected(self):
        tiles = [cv2.resize(cv2.imread(p), (320, 320)) for p in RAW[:4]]
        mosaic = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mosaic.jpg'
            cv2.imwrite(str(path), mosaic)
            self.assertTrue(is_mosaic(seam_scores(path)))

    @unittest.skipUnless(Path('data/processed/rust/manifest.json').is_file(), 'rust manifest not prepared')
    def test_manifest_split_has_no_shared_source_or_image(self):
        records = json.loads(Path('data/processed/rust/manifest.json').read_text(encoding='utf-8'))['records']
        kept = [r for r in records if r['split'] != 'excluded_mosaic']
        for key in ('source', 'sha256'):
            splits = {}
            for r in kept:
                splits.setdefault(r[key], set()).add(r['split'])
            self.assertFalse([k for k, v in splits.items() if len(v) > 1], f'{key} crosses splits')


class CorrosionFindingTests(unittest.TestCase):
    def run_for(self, corrosion, severe):
        grid = [[0.0] * 9 for _ in range(9)]
        grid[2][6] = grid[2][7] = grid[3][7] = 1.0
        return {'corrosion': corrosion, 'severe': severe, 'corrosion_probability': .9, 'severe_probability': .8 if severe else .1,
                'activation_map': grid}

    def test_no_finding_without_corrosion(self):
        self.assertEqual(_corrosion_defects(self.run_for(False, False), (288, 288), None, None), [])
        self.assertEqual(_corrosion_defects(None, (288, 288), None, None), [])

    def test_severe_corrosion_is_high_but_never_critical(self):
        defects = _corrosion_defects(self.run_for(True, True), (288, 288), None, None)
        self.assertEqual(defects[0]['label'], 'severe_corrosion')
        self.assertEqual(defects[0]['region_box_px'], [192.0, 64.0, 256.0, 128.0])   # activation map cells -> image region
        _apply_severity(defects, None, (288, 288))
        self.assertEqual(defects[0]['severity']['level'], 'high')
        mild = _corrosion_defects(self.run_for(True, False), (288, 288), None, None)
        _apply_severity(mild, None, (288, 288))
        self.assertEqual(mild[0]['label'], 'corrosion')
        self.assertIn(mild[0]['severity']['level'], ('medium', 'high'))


class CorrosionRuntimeTests(unittest.TestCase):
    def test_refuses_missing_or_unevaluated_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / 'rust.pt'
            classifier = rust_classifier.RustClassifier(Path(folder) / 'missing.pt')
            with self.assertRaises(rust_classifier.RustUnavailable):
                classifier.detect(COLOUR)
            fake.write_bytes(b'not the evaluated model')
            evaluation = Path(folder) / 'evaluation.json'
            evaluation.write_text(json.dumps({'artifact_sha256': '0' * 64}))
            with patch.object(rust_classifier, 'EVALUATION', evaluation):
                with self.assertRaises(rust_classifier.RustUnavailable):
                    rust_classifier.RustClassifier(fake).detect(COLOUR)


class CorrosionDomainTests(unittest.TestCase):
    def test_greyscale_images_are_skipped_not_guessed(self):
        result = rust_classifier.RustClassifier(Path('missing.pt')).detect(np.full((64, 64, 3), 120, np.uint8))
        self.assertTrue(result['skipped'])
        self.assertFalse(result['corrosion'])

    @unittest.skipUnless(Path('models/rust/domain_thresholds.json').is_file() and Path('models/rust/rust_classifier.pt').is_file(), 'rust model not trained')
    def test_metal_plate_uses_its_calibrated_threshold(self):
        classifier = rust_classifier.RustClassifier()
        plate = cv2.imread(sorted(glob.glob('data/raw/MPDD/metal_plate/train/good/*.png'))[0])
        general, calibrated = classifier.detect(plate), classifier.detect(plate, domain='mpdd_metal_plate')
        self.assertEqual(calibrated['thresholds']['domain'], 'mpdd_metal_plate')
        self.assertGreater(calibrated['thresholds']['corrosion'], general['thresholds']['corrosion'])
        self.assertFalse(calibrated['corrosion'])        # a normal training plate never exceeds the max-of-normals threshold
        self.assertFalse(calibrated['severe'])           # severe head is not trusted on plates


if __name__ == '__main__':
    unittest.main()
