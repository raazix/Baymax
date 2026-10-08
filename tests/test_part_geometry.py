import glob
import json
import math
import unittest

import cv2
import numpy as np

from vision.part_geometry import locate_part, polar_position, scale_mm_per_px, anomaly_region


def synthetic_part(cx=300, cy=260, a=210, b=180, angle=20, size=(560, 600)):
    """Grey background, bright metal disc (optionally tilted into an ellipse), dark concentric interior."""
    image = np.full((*size, 3), 128, np.uint8)
    cv2.ellipse(image, ((cx, cy), (2 * a, 2 * b), angle), (215, 215, 215), -1)
    cv2.ellipse(image, ((cx, cy), (a, b), angle), (25, 25, 25), -1)
    cv2.ellipse(image, ((cx, cy), (a * .35, b * .35), angle), (190, 190, 190), -1)
    noise = np.random.default_rng(0).normal(0, 4, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


class PartGeometryTests(unittest.TestCase):
    def test_synthetic_part_centre_and_rim_are_accurate(self):
        for a, b, angle in ((200, 200, 0), (210, 170, 25)):
            geometry = locate_part(synthetic_part(a=a, b=b, angle=angle))
            self.assertIsNotNone(geometry)
            cx, cy = geometry['center_px']
            self.assertLess(math.hypot(cx - 300, cy - 260), 4)
            self.assertAlmostEqual(max(geometry['semi_axes_px']), a, delta=a * .04)
            self.assertAlmostEqual(min(geometry['semi_axes_px']), b, delta=b * .04)

    def test_real_castings_located_and_flat_steel_rejected(self):
        with open('data/processed/casting/manifest.json', encoding='utf-8') as handle:
            records = json.load(handle)['records']
        test = [r['path'] for r in records if r['split'] == 'test'][:60]
        located = sum(locate_part(cv2.imread(p)) is not None for p in test)
        self.assertGreaterEqual(located / len(test), .85)
        steel = sorted(glob.glob('data/processed/neu-det/images/test/*.jpg'))[:40]
        self.assertEqual(sum(locate_part(cv2.imread(p)) is not None for p in steel), 0)
        self.assertIsNone(locate_part(np.full((480, 640, 3), 128, np.uint8)))

    def test_polar_conventions(self):
        geometry = {'center_px': [100, 100], 'semi_axes_px': [80, 80], 'angle_deg': 0}
        right = polar_position(geometry, 180, 100)
        self.assertEqual((right['theta_deg'], right['r_fraction'], right['zone']), (0, 1, 'rim'))
        up = polar_position(geometry, 100, 60)          # image y grows downward, so up is 90 degrees
        self.assertEqual((up['theta_deg'], up['r_fraction'], up['zone']), (90, .5, 'mid-radius'))
        self.assertEqual(polar_position(geometry, 100, 110)['zone'], 'bore / hub')
        tilted = {'center_px': [100, 100], 'semi_axes_px': [80, 40], 'angle_deg': 0}
        self.assertEqual(polar_position(tilted, 100, 140)['r_fraction'], 1)  # rim along the short axis is still fraction 1

    def test_scale_only_from_operator_input(self):
        geometry = {'center_px': [0, 0], 'semi_axes_px': [200, 150], 'angle_deg': 0}
        self.assertEqual(scale_mm_per_px(geometry), (None, None))
        self.assertAlmostEqual(scale_mm_per_px(geometry, part_diameter_mm=100)[0], .25)
        self.assertEqual(scale_mm_per_px(None, mm_per_px=.1)[0], .1)
        self.assertEqual(scale_mm_per_px(None, part_diameter_mm=100), (None, None))

    def test_anomaly_region_from_grid(self):
        grid = np.ones((28, 28))
        grid[5:8, 20:24] = 3.0
        grid[20, 3] = 2.6                       # separate weaker blob is not merged into the peak region
        region = anomaly_region(grid, 2.0, 560, 560)
        self.assertEqual(region['cells'], 12)
        self.assertEqual(region['region_box_px'], [400.0, 100.0, 480.0, 160.0])
        self.assertEqual(region['length_px'], 80.0)
        self.assertIsNone(anomaly_region(np.ones((28, 28)), 2.0, 560, 560))


if __name__ == '__main__':
    unittest.main()
