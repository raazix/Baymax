"""Evidence-pinned analytics composition and historical replay contracts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analytics import service
from analytics.demo import analyze as legacy_analyze


class AnalyticsServiceTests(unittest.TestCase):
    def setUp(self):
        self.telemetry = dict(temperature_c=758, pressure_bar=113, vibration_mm_s=4.1, machine_speed_rpm=1200)

    def test_legacy_pin_preserves_original_results_with_trained_artifacts_available(self):
        expected = legacy_analyze(self.telemetry, 42)
        with patch('analytics.service.model_versions', return_value={'analytics': 'trained-synthetic-v1'}):
            actual = service.analyze(self.telemetry, 42, {'analytics': 'demo-heuristic-v1'})
        for key, value in expected.items():
            self.assertEqual(actual[key], value)
        self.assertEqual(actual['model_versions'], {'analytics': 'demo-heuristic-v1'})
        self.assertIn('legacy', actual['availability_note'])

    def test_missing_or_changed_pinned_artifact_fails_without_heuristic_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.bin'
            path.write_bytes(b'original')
            artifacts = {'plsr': path}
            versions = {'analytics': 'trained-synthetic-v1', 'plsr': hashlib.sha256(b'original').hexdigest()}
            with patch.dict(service.ARTIFACTS, artifacts, clear=True), patch('analytics.service.demo_analyze') as fallback:
                path.write_bytes(b'replacement')
                with self.assertRaisesRegex(service.AnalyticsUnavailable, 'changed'):
                    service.analyze(self.telemetry, 42, versions)
                path.unlink()
                with self.assertRaisesRegex(service.AnalyticsUnavailable, 'unavailable'):
                    service.analyze(self.telemetry, 42, versions)
                fallback.assert_not_called()

    def test_unknown_version_is_rejected(self):
        with self.assertRaisesRegex(service.AnalyticsUnavailable, 'Unknown'):
            service.analyze(self.telemetry, 42, {'analytics': 'unsupported-version'})

    def test_trained_composition_is_json_safe_and_preserves_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = {}
            for name in ('xgboost_rca', 'rca_metadata', 'plsr'):
                path = Path(directory) / name
                path.write_bytes(name.encode())
                artifacts[name] = path
            rca = {'hypothesis': 'thermal_process_drift', 'causality_established': False,
                   'provenance': {'model_sha256': hashlib.sha256(b'xgboost_rca').hexdigest()}}
            forecast = {'forecast': {'risk': .65, 'production_validated': False,
                                    'artifact_sha256': hashlib.sha256(b'plsr').hexdigest()},
                        'uncertainty': {'mean': .64, 'interval_95': [.5, .8]}}
            with patch.dict(service.ARTIFACTS, artifacts, clear=True), \
                    patch('analytics.xgboost_rca.analyze_rca', return_value=rca) as classifier, \
                    patch('analytics.plsr_forecast.analyze_forecast', return_value=forecast) as regressor:
                versions = service.model_versions()
                actual = service.analyze(self.telemetry, 31, versions)
                classifier.assert_called_once_with(self.telemetry)
                regressor.assert_called_once_with(self.telemetry, 31)
            self.assertEqual(actual['model_versions'], versions)
            self.assertEqual(actual['rca'], rca)
            self.assertEqual(actual['forecast'], forecast['forecast'])
            self.assertFalse(actual['production_validated'])
            self.assertEqual(json.loads(json.dumps(actual, allow_nan=False)), actual)

    def test_override_directory_pins_actual_configured_rca_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.ubj').write_bytes(b'override-model')
            (root / 'metadata.json').write_bytes(b'override-metadata')
            plsr = root / 'forecast.joblib'
            plsr.write_bytes(b'forecast')
            with patch.dict(os.environ, {'LINEGUARD_RCA_MODEL_DIR': directory}), \
                    patch.dict(service.ARTIFACTS, {'plsr': plsr}, clear=True):
                versions = service.model_versions()
                self.assertEqual(versions['xgboost_rca'], hashlib.sha256(b'override-model').hexdigest())
                self.assertEqual(versions['rca_metadata'], hashlib.sha256(b'override-metadata').hexdigest())
                self.assertEqual(service.artifact_paths()['xgboost_rca'], root.resolve() / 'model.ubj')

    def test_loaded_model_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = {}
            for name in ('xgboost_rca', 'rca_metadata', 'plsr'):
                path = Path(directory) / name
                path.write_bytes(name.encode())
                artifacts[name] = path
            with patch.dict(service.ARTIFACTS, artifacts, clear=True), \
                    patch('analytics.xgboost_rca.analyze_rca', return_value={'provenance': {'model_sha256': 'stale'}}), \
                    patch('analytics.plsr_forecast.analyze_forecast', return_value={'forecast': {'artifact_sha256': 'stale'}}):
                with self.assertRaisesRegex(service.AnalyticsUnavailable, 'Loaded models'):
                    service.analyze(self.telemetry, 42, service.model_versions())

    def test_real_artifact_composition_when_available(self):
        versions = service.model_versions()
        if versions['analytics'] != 'trained-synthetic-v1':
            self.skipTest('RCA artifacts still being trained')
        actual = service.analyze(self.telemetry, 42, versions)
        self.assertEqual(actual['forecast']['method'], 'trained PLSR on synthetic ordered lots')
        self.assertIn('TreeSHAP', actual['rca']['method'])
        self.assertEqual(json.loads(json.dumps(actual, allow_nan=False)), actual)


if __name__ == '__main__':
    unittest.main()
