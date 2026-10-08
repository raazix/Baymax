import unittest
import hashlib
from io import BytesIO
import os
import tempfile
import joblib
from pathlib import Path
import numpy as np
from analytics.plsr_forecast import analyze_forecast, feature_row, load_artifact, ForecastModelUnavailable
from analytics.pcr_forecast import predict_benchmark
from analytics.monte_carlo import simulate
from scripts.train_forecast import synthetic_lots


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.telemetry = dict(temperature_c=715, pressure_bar=100, vibration_mm_s=3, machine_speed_rpm=1200)

    def test_temporal_feature_order_and_missing_history(self):
        earlier = dict(self.telemetry, temperature_c=690)
        latest = dict(self.telemetry, temperature_c=700)
        row, note = feature_row({**self.telemetry, 'history': [earlier, latest]})
        self.assertEqual(row[0], 715)
        self.assertEqual(row[4], 700)
        self.assertEqual(row[8], 695)
        self.assertEqual(note, 'observed prior-lot history')
        self.assertIn('persistence', feature_row(self.telemetry)[1])

    def test_bootstrap_matches_seed_and_clips_coherently(self):
        residuals = np.linspace(-.3, .3, 40)
        result = simulate(.8, residuals, seed=42)
        expected = np.clip(.8 + np.random.default_rng(42).choice(residuals, 10000, replace=True), 0, 1)
        self.assertEqual(result, simulate(.8, residuals, seed=42))
        self.assertAlmostEqual(result['mean'], expected.mean())
        self.assertEqual(result['breach_probability'], float(np.mean(expected > .5)))
        np.testing.assert_array_equal(result['interval_95'], np.quantile(expected, [.025, .975]))

    def test_artifact_and_held_out_metrics(self):
        artifact = load_artifact()
        self.assertLess(artifact['metrics']['test_mae'], artifact['metrics']['persistence_baseline_test_mae'])
        x, _ = synthetic_lots()
        end = artifact['provenance']['split']['train']
        np.testing.assert_allclose(artifact['model'].steps[0][1].mean_, x[:end].mean(axis=0))
        self.assertEqual(len(artifact['residuals']), 360)

    def test_inference_and_benchmark_labels(self):
        result = analyze_forecast(self.telemetry, 123)
        self.assertEqual(result, analyze_forecast(self.telemetry, 123))
        self.assertFalse(result['forecast']['production_validated'])
        self.assertFalse(result['forecast']['calibrated_event_probability'])
        self.assertEqual(predict_benchmark(self.telemetry)['role'], 'benchmark only')
        self.assertTrue(0 <= result['forecast']['risk'] <= 1)

    def test_missing_model_and_invalid_telemetry_fail_explicitly(self):
        with self.assertRaises(ForecastModelUnavailable):
            load_artifact(Path('models/nonexistent-test-model.joblib'))
        with self.assertRaises(ValueError):
            analyze_forecast(dict(self.telemetry, temperature_c=float('nan')), 42)
        with self.assertRaises(ValueError):
            feature_row({'temperature_c': 700})
        with self.assertRaises(ValueError):
            simulate(.5, [0], 42)

    def test_same_mtime_artifact_replacement_loads_correct_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forecast.joblib'
            buffer = BytesIO()
            joblib.dump({'version': 'first'}, buffer)
            first_bytes = buffer.getvalue()
            path.write_bytes(first_bytes)
            original_time = path.stat().st_mtime_ns
            first = load_artifact(path)
            buffer = BytesIO()
            joblib.dump({'version': 'other'}, buffer)
            second_bytes = buffer.getvalue()
            path.write_bytes(second_bytes)
            os.utime(path, ns=(original_time, original_time))
            second = load_artifact(path)
            self.assertEqual(path.stat().st_mtime_ns, original_time)
            self.assertEqual(first['version'], 'first')
            self.assertEqual(second['version'], 'other')
            self.assertEqual(first['_loaded_sha256'], hashlib.sha256(first_bytes).hexdigest())
            self.assertEqual(second['_loaded_sha256'], hashlib.sha256(second_bytes).hexdigest())


if __name__ == '__main__':
    unittest.main()
