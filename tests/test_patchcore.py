import hashlib
import builtins
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.models import resnet18

from vision.patchcore_anomaly import (
    ROOT, FrozenFeatures, PatchCoreDetector, PatchCoreUnavailable,
    extract_patch_features, greedy_coreset, nearest_patch_distances, preprocess_frames,
)


class PatchCoreTests(unittest.TestCase):
    def test_module_and_status_boot_without_ml_imports(self):
        import vision.patchcore_anomaly as module
        original_import = builtins.__import__
        def restricted_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'torchvision'):
                raise ImportError('ML dependencies intentionally unavailable')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=restricted_import):
            spec = importlib.util.spec_from_file_location('patchcore_without_ml', module.__file__)
            independent = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(independent)
            self.assertFalse(independent.PatchCoreDetector('missing.pt').status()['loaded'])

    def test_feature_shape_frozen_and_shared_preprocessing(self):
        with patch('torch.hub.download_url_to_file', side_effect=AssertionError('No downloads allowed')):
            extractor = FrozenFeatures(resnet18(weights=None))
            frames = [np.full((41, 73, 3), [0, 0, 255], dtype=np.uint8)]
            batch = preprocess_frames(frames)
            self.assertEqual(tuple(batch.shape), (1, 3, 224, 224))
            self.assertAlmostEqual(float(batch[0, 0, 0, 0]), (1 - .485) / .229, places=5)
            self.assertAlmostEqual(float(batch[0, 2, 0, 0]), - .406 / .225, places=5)
            features = extract_patch_features(extractor, frames)
            self.assertEqual(tuple(features.shape), (1, 784, 384))
            self.assertFalse(features.requires_grad)
            self.assertTrue(all(not parameter.requires_grad for parameter in extractor.parameters()))
            self.assertFalse(extractor.training)

    def test_chunked_nearest_neighbor_matches_reference_and_clamps(self):
        generator = torch.Generator().manual_seed(15)
        features = torch.randn(19, 17, generator=generator)
        bank = torch.cat([features[:2], torch.randn(23, 17, generator=generator)])
        expected = torch.cdist(features, bank).amin(1)
        actual = nearest_patch_distances(features, bank, chunk_size=4, bank_chunk_size=6)
        torch.testing.assert_close(actual[2:], expected[2:], rtol=1e-5, atol=1e-5)
        self.assertTrue(bool((actual >= 0).all()))
        self.assertLess(float(actual[:2].max()), .002)

    def test_greedy_coreset_reproducibility_and_coverage(self):
        features = torch.linspace(0, 10, 101).reshape(-1, 1)
        bank, info = greedy_coreset(features, max_points=8, projection_dim=4, seed=17)
        bank2, info2 = greedy_coreset(features, max_points=8, projection_dim=4, seed=17)
        torch.testing.assert_close(bank, bank2)
        self.assertEqual(info, info2)
        self.assertEqual(len(torch.unique(bank)), 8)
        coverage = nearest_patch_distances(features, bank).max()
        first_coverage = nearest_patch_distances(features, bank[:1]).max()
        self.assertLess(float(coverage), float(first_coverage))
        self.assertLessEqual(float(coverage), 1.3)
        _, bounded = greedy_coreset(features, max_points=8, projection_dim=4, seed=17, candidate_limit=20)
        self.assertEqual(bounded['candidate_patches'], 20)
        self.assertTrue(bounded['candidate_pool_subsampled'])

    def test_missing_artifact_fail_closed_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = PatchCoreDetector(Path(directory) / 'missing.pt')
            self.assertFalse(detector.status()['configured'])
            with patch('vision.patchcore_anomaly.load_feature_extractor') as loader:
                with self.assertRaises(PatchCoreUnavailable):
                    detector.detect(np.zeros((20, 20, 3), np.uint8))
                loader.assert_not_called()

    def test_hash_mismatch_rejected_before_backbone_load(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as directory:
            root = Path(directory)
            backbone = root / 'backbone.pth'
            backbone.write_bytes(b'local backbone fixture')
            artifact = root / 'proxy.pt'
            torch.save({'format_version': 1, 'bank': torch.ones(4, 384), 'threshold': 1.,
                        'backbone_path': str(backbone.relative_to(ROOT)), 'backbone_sha256': '0' * 64,
                        'metadata': {'method': 'compact_patchcore_resnet18', 'data_source': 'casting_proxy', 'brake_disc_validated': False}}, artifact)
            with patch('vision.patchcore_anomaly.load_feature_extractor') as loader:
                with self.assertRaisesRegex(PatchCoreUnavailable, 'hash'):
                    PatchCoreDetector(artifact).detect(np.zeros((20, 20, 3), np.uint8))
                loader.assert_not_called()

    def test_invalid_bank_threshold_and_format_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'proxy.pt'
            for changes in [{'bank': torch.zeros(2, 383)}, {'bank': torch.full((2, 384), float('nan'))},
                            {'threshold': float('inf')}, {'threshold': 0.}, {'format_version': 2}]:
                payload = {'format_version': 1, 'bank': torch.ones(2, 384), 'threshold': 1.,
                           'metadata': {'method': 'compact_patchcore_resnet18', 'data_source': 'casting_proxy', 'brake_disc_validated': False},
                           **changes}
                torch.save(payload, artifact)
                with self.subTest(changes=list(changes)):
                    with self.assertRaises(PatchCoreUnavailable):
                        PatchCoreDetector(artifact).detect(np.zeros((20, 20, 3), np.uint8))

    def test_numeric_shared_gpu_setting_is_normalized_for_torch(self):
        with patch.dict(os.environ, {'LINEGUARD_MODEL_DEVICE': '0'}):
            self.assertEqual(PatchCoreDetector('unused.pt').device, 'cuda:0')

    def test_detection_score_grid_and_nonprobability_contract(self):
        detector = PatchCoreDetector('unused.pt')
        detector.extractor = object()
        detector.bank = torch.zeros(1, 384)
        detector.threshold = 1.
        detector.digest = 'fixture'
        detector.metadata = {'data_source': 'casting_proxy'}
        features = torch.zeros(1, 784, 384)
        features[0, 20, 0] = 2
        with patch('vision.patchcore_anomaly.extract_patch_features', return_value=features):
            result = detector.detect(np.zeros((20, 20, 3), np.uint8))
        self.assertEqual(result['label'], 'ANOMALY_UNCLASSIFIED')
        self.assertEqual(result['anomaly_score'], 2.)
        self.assertEqual(np.asarray(result['anomaly_grid']).shape, (28, 28))
        self.assertFalse(result['brake_disc_validated'])
        self.assertIsNone(result['metrology'])
        self.assertIn('not a probability', result['score_units'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
