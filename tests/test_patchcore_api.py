import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from apps.api import main
from database.repository import Repository
from vision.patchcore_anomaly import PatchCoreUnavailable


class PatchCoreAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = main.repo
        main.repo = Repository('sqlite:///' + (Path(self.directory.name) / 'test.db').as_posix())
        self.environment = patch.dict('os.environ', {'LINEGUARD_API_TOKEN': ''})
        self.environment.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        main.repo.engine.dispose()
        main.repo = self.previous
        self.environment.stop()
        self.directory.cleanup()

    def upload(self, blurred=False):
        rng = np.random.default_rng(42)
        frame = np.full((64,64,3),100,dtype=np.uint8) if blurred else rng.integers(20,220,(64,64,3),dtype=np.uint8)
        encoded = cv2.imencode('.png', frame)[1].tobytes()
        result = self.client.post('/api/frames', content=encoded, headers={'Content-Type':'image/png'})
        self.assertEqual(result.status_code,201)
        return result.json()

    def test_quality_and_missing_frame_do_not_invoke_model(self):
        with patch.object(main, 'patchcore_detector') as detector:
            rejected = self.upload(blurred=True)
            self.assertEqual(self.client.post('/api/proxy/frames/'+rejected['id']+'/anomaly').status_code,422)
            self.assertEqual(self.client.post('/api/proxy/frames/missing/anomaly').status_code,404)
            detector.detect.assert_not_called()

    def test_unavailable_is_fail_closed_without_invented_predictions(self):
        accepted = self.upload()
        detector = Mock()
        detector.detect.side_effect = PatchCoreUnavailable('No trained normal bank')
        with patch.object(main,'patchcore_detector',detector):
            response = self.client.post('/api/proxy/frames/'+accepted['id']+'/anomaly')
        self.assertEqual(response.status_code,503)
        self.assertNotIn('anomaly_score',response.json())

    def test_persisted_anomaly_pins_frame_and_artifact_without_metrology(self):
        accepted = self.upload()
        detector = Mock()
        detector.detect.return_value = {'source':'trained_casting_patchcore_proxy','label':'ANOMALY_UNCLASSIFIED',
            'anomaly_score':3.,'threshold':2.,'anomaly_grid':[[3.]*28 for _ in range(28)],
            'artifact_sha256':'a'*64,'model_sha256':'a'*64,'brake_disc_validated':False,'metrology':None}
        with patch.object(main,'patchcore_detector',detector):
            response = self.client.post('/api/proxy/frames/'+accepted['id']+'/anomaly')
        self.assertEqual(response.status_code,200)
        result = response.json()
        self.assertEqual(result['image_sha256'],accepted['sha256'])
        self.assertEqual(self.client.get('/api/proxy/runs/'+result['id']).json(),result)
        self.assertFalse(result['brake_disc_validated'])
        self.assertIsNone(result['metrology'])
        self.assertIn('no cross-model fusion',result['domain_note'])

if __name__ == '__main__':
    unittest.main()
