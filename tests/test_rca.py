import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analytics.xgboost_rca import MODEL_DIR, RCAModelUnavailable, _load, analyze_rca
from scripts.train_xgboost import generate_history


class RCATests(unittest.TestCase):
    def test_history_reproducibility_and_future_lot_split(self):
        x, y, lots = generate_history()
        x2, y2, lots2 = generate_history()
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(y, y2)
        self.assertFalse(set(lots[lots < 400]) & set(lots[lots >= 400]))
        self.assertEqual(len(set(lots[lots >= 400])), 100)

    def test_missing_model_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'LINEGUARD_RCA_MODEL_DIR': directory}):
                with self.assertRaises(RCAModelUnavailable):
                    analyze_rca(self.telemetry())

    @staticmethod
    def telemetry():
        return dict(temperature_c=760, pressure_bar=103, vibration_mm_s=2.2, machine_speed_rpm=1200)

    def test_real_artifact_explanation_additivity(self):
        result = analyze_rca(self.telemetry())
        self.assertEqual(result['hypothesis'], 'thermal_process_drift')
        self.assertFalse(result['causality_established'])
        self.assertFalse(result['confidence_calibrated'])
        self.assertEqual(result['provenance']['dataset_status'], 'synthetic')
        self.assertAlmostEqual(sum(result['class_probabilities'].values()), 1, places=5)
        self.assertAlmostEqual(result['base_value'] + sum(row['contribution'] for row in result['feature_contributions']),
                               result['prediction_margin'], places=5)
        self.assertEqual(result['provenance']['split']['test_lots'], [400, 499])

    def test_invalid_telemetry(self):
        for value in [float('nan'), float('inf'), 'invalid']:
            telemetry = self.telemetry()
            telemetry['temperature_c'] = value
            with self.assertRaises(ValueError):
                analyze_rca(telemetry)

    def test_cache_invalidates_metadata_and_rejects_tampered_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ['model.ubj', 'metadata.json']:
                shutil.copyfile(MODEL_DIR / filename, root / filename)
            with patch.dict(os.environ, {'LINEGUARD_RCA_MODEL_DIR': directory}):
                first = analyze_rca(self.telemetry())
                metadata = json.loads((root / 'metadata.json').read_text())
                metadata['model_id'] = 'replacement-provenance'
                (root / 'metadata.json').write_text(json.dumps(metadata))
                second = analyze_rca(self.telemetry())
                self.assertNotEqual(first['provenance']['model_id'], second['provenance']['model_id'])
                with (root / 'model.ubj').open('ab') as model:
                    model.write(b'tampered')
                with self.assertRaises(RCAModelUnavailable):
                    analyze_rca(self.telemetry())


if __name__ == '__main__':
    unittest.main()
