import glob
import unittest
import cv2
import numpy as np
from vision.quality_gate import check_frame, check_frame_open_domain
from vision.patchcore_anomaly import PatchCoreDetector, GRID_SIZE


def casting(index=0):
    paths = sorted(glob.glob('data/raw/casting_512x512/ok_front/*'))
    return cv2.imread(paths[index])


class OpenDomainGateTests(unittest.TestCase):
    def test_dataset_sized_image_is_unchanged(self):
        frame = casting()
        self.assertEqual(check_frame_open_domain(frame, 'casting_proxy'), check_frame(frame, 'casting_proxy'))

    def test_other_lighting_and_size_are_waived_not_rejected(self):
        frame = cv2.resize(casting(), (715, 429))
        darker = np.clip(frame.astype(np.float32) * .6, 0, 255).astype(np.uint8)
        strict = check_frame(darker, 'casting_proxy')
        self.assertFalse(strict['passed'])
        relaxed = check_frame_open_domain(darker, 'casting_proxy')
        self.assertTrue(relaxed['passed'])
        self.assertIn('dimensions_outside_calibrated_proxy_domain', relaxed['waived_checks'])
        self.assertEqual(relaxed['rejection_reasons'], [])

    def test_real_failures_are_never_waived(self):
        frame = casting()
        cases = {'blank': np.full_like(frame, 128), 'blurry': cv2.GaussianBlur(frame, (21, 21), 8),
                 'black': np.zeros_like(frame), 'blown_out': np.full_like(frame, 255)}
        for name, image in cases.items():
            for size in (None, (715, 429)):
                candidate = image if size is None else cv2.resize(image, size)
                self.assertFalse(check_frame_open_domain(candidate, 'casting_proxy')['passed'], (name, size))

    def test_camera_profile_untouched(self):
        frame = np.random.default_rng(1).integers(20, 230, (480, 640, 3), dtype=np.uint8)
        self.assertEqual(check_frame_open_domain(frame, 'camera'), check_frame(frame, 'camera'))


class SlicedPatchCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = PatchCoreDetector()

    def test_square_path_unchanged(self):
        result = self.detector.detect(casting())
        self.assertIsNone(result['tiling'])
        self.assertEqual(result['grid_shape'], [GRID_SIZE, GRID_SIZE])

    def test_wide_and_tall_images_keep_aspect_ratio(self):
        a, b = casting(0), casting(1)
        wide = self.detector.detect(np.hstack([a, b]))
        self.assertEqual(wide['grid_shape'], [GRID_SIZE, 2 * GRID_SIZE])
        self.assertGreaterEqual(wide['tiling']['crops'], 2)
        tall = self.detector.detect(np.vstack([a, b]))
        self.assertEqual(tall['grid_shape'], [2 * GRID_SIZE, GRID_SIZE])
        singles = max(self.detector.detect(a)['anomaly_score'], self.detector.detect(b)['anomaly_score'])
        self.assertGreaterEqual(wide['anomaly_score'] + 1e-6, singles - 1e-6)

    def test_extreme_aspect_ratio_is_refused(self):
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((40, 4000, 3), np.uint8) + 100)


if __name__ == '__main__':
    unittest.main()
