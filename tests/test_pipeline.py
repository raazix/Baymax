import hashlib
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from core.schemas import Detection, ReplayRequest
from core.replay import create_replay, render_svg
from vision.calibration import mm_per_pixel
from vision.severity import rate
from database.repository import Repository
from apps.api import main

class PipelineTests(unittest.TestCase):
    def test_calibration_and_severity_independent_of_confidence(self):
        self.assertAlmostEqual(mm_per_pixel(161), 10 / 161)
        with self.assertRaises(ValueError): mm_per_pixel(0)
        for confidence in [.01, .99]:
            result = rate({'label': 'surface_crack', 'length_mm': 3.75, 'zone': 'vane_root', 'confidence': confidence})
            self.assertEqual(result['level'], 'critical')

    def test_replay_hash_and_seed(self):
        a = create_replay(ReplayRequest())
        b = create_replay(ReplayRequest())
        self.assertEqual(a['analytics'], b['analytics'])
        self.assertEqual(a['image_sha256'], hashlib.sha256(render_svg(a).encode()).hexdigest())
        self.assertFalse(a['analytics']['rca'].get('confidence_calibrated', False))
        self.assertFalse(a['analytics']['rca']['causality_established'])
        self.assertEqual(a['defects'][0]['severity']['level'], 'critical')

    def test_quality_rejection_stops_analytics(self):
        result = create_replay(ReplayRequest(scenario='blurred'))
        self.assertIsNone(result['analytics'])
        self.assertEqual(result['disposition'], 'recapture')

    def test_decision_and_verification_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = main.repo
            main.repo = Repository('sqlite:///' + (Path(directory) / 'test.db').as_posix())
            try:
                with TestClient(main.app) as client:
                    result = client.post('/api/replay', json={}).json()
                    identifier = result['id']
                    self.assertEqual(client.post(f'/api/inspections/{identifier}/verification', json={'inspection_ids': [identifier] * 20}).status_code, 409)
                    self.assertEqual(client.post(f'/api/inspections/{identifier}/decision', json={'decision': 'approve', 'engineer': 'Engineer A'}).status_code, 200)
                    self.assertEqual(client.post(f'/api/inspections/{identifier}/decision', json={'decision': 'reject', 'engineer': 'Engineer A'}).status_code, 409)
                    self.assertEqual(client.post(f'/api/inspections/{identifier}/verification', json={'inspection_ids': [identifier] * 20}).status_code, 422)
                    evidence = [client.post('/api/replay', json={'scenario': 'normal'}).json()['id'] for _ in range(20)]
                    verified = client.post(f'/api/inspections/{identifier}/verification', json={'inspection_ids': evidence})
                    self.assertEqual(verified.status_code, 200)
                    self.assertEqual(verified.json()['verification']['defective'], 0)
                    self.assertFalse(verified.json()['verification']['production_fix_verified'])
                    self.assertEqual(client.get(f'/api/inspections/{identifier}').json()['action']['status'], 'verification_recorded')
                    self.assertEqual(client.get('/api/inspections/missing').status_code, 404)
                    self.assertEqual(client.post('/api/replay', json={'scenario': 'made-up'}).status_code, 422)
            finally:
                main.repo.engine.dispose()
                main.repo = previous

if __name__ == '__main__': unittest.main()
