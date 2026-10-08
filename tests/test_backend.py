import hashlib
import os
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import select
from apps.api import main
from analytics.worker import process_one
from database.repository import Repository
from database.models import Inspection, Part, Telemetry, RiskAndRCA, CorrectiveAction

class BackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = main.repo
        main.repo = Repository('sqlite:///' + (Path(self.directory.name) / 'test.db').as_posix())
        self.client = TestClient(main.app)
        self.auth = patch.dict(os.environ, {'LINEGUARD_API_TOKEN': ''})
        self.auth.start()

    def tearDown(self):
        self.auth.stop()
        self.client.close()
        main.repo.engine.dispose()
        main.repo = self.previous
        self.directory.cleanup()

    def replay(self, **kwargs):
        response = self.client.post('/api/replay', json=kwargs)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_durable_deferred_job_and_projections(self):
        record = self.replay(analytics_mode='deferred')
        self.assertEqual(record['action']['status'], 'awaiting_analytics')
        self.assertEqual(self.client.post(f"/api/inspections/{record['id']}/decision", json={'decision':'approve','engineer':'Engineer A'}).status_code, 409)
        reopened = Repository(str(main.repo.engine.url))
        try:
            self.assertEqual(reopened.get_job(record['analytics_job_id'])['status'], 'queued')
            self.assertTrue(process_one(reopened))
            self.assertFalse(process_one(reopened))
            result = reopened.get(record['id'])
            self.assertEqual(result['action']['status'], 'pending')
            self.assertEqual(reopened.get_job(record['analytics_job_id'])['status'], 'completed')
            self.assertTrue(reopened.verify_audit(record['id'])['valid'])
            with reopened.engine.connect() as conn:
                for model in [Part, Telemetry, RiskAndRCA, CorrectiveAction]:
                    self.assertIsNotNone(conn.execute(select(model)).first())
        finally: reopened.engine.dispose()

    def test_failed_job_can_retry_without_duplicate_audit(self):
        record = self.replay(analytics_mode='deferred')
        with patch('analytics.worker.analyze', side_effect=RuntimeError('test failure')), self.assertLogs('analytics.worker', level='ERROR'):
            self.assertTrue(process_one(main.repo))
        identifier = record['analytics_job_id']
        self.assertEqual(main.repo.get_job(identifier)['status'], 'failed')
        self.assertEqual(self.client.post(f'/api/jobs/{identifier}/retry').status_code, 200)
        self.assertTrue(process_one(main.repo))
        self.assertEqual(self.client.post(f'/api/jobs/{identifier}/retry').status_code, 409)
        result = main.repo.get(record['id'])
        self.assertEqual(sum(e['event']=='analytics_completed' for e in result['audit']), 1)

    def test_worker_crash_recovery_keeps_a_single_completed_result(self):
        record = self.replay(analytics_mode='deferred')
        identifier = record['analytics_job_id']
        self.assertEqual(main.repo.claim_job(), identifier)
        self.assertEqual(main.repo.get_job(identifier)['status'], 'running')
        main.repo.recover_jobs()
        self.assertTrue(process_one(main.repo))
        # Simulate a crash after committing results but before marking job complete.
        from database.models import AnalyticsJob
        from sqlalchemy.orm import Session
        with Session(main.repo.engine) as session, session.begin():
            session.get(AnalyticsJob, identifier).status = 'running'
        main.repo.recover_jobs()
        self.assertTrue(process_one(main.repo))
        result = main.repo.get(record['id'])
        self.assertEqual(sum(e['event']=='analytics_completed' for e in result['audit']), 1)
        self.assertTrue(main.repo.verify_audit(record['id'])['valid'])

    def test_oversized_and_invalid_images_rejected_before_decode(self):
        self.assertEqual(self.client.post('/api/frames', content=b'not an image').status_code, 422)
        huge_png = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\x0dIHDR' + struct.pack('>II', 65535, 65535)
        self.assertEqual(self.client.post('/api/frames', content=huge_png).status_code, 413)
        self.assertEqual(self.client.post('/api/frames', content=b'x'*(8*1024*1024+1)).status_code, 413)

    def test_frames_keep_exact_bytes_and_fail_closed_without_model(self):
        frame = np.random.default_rng(1).integers(10, 230, (128,128,3),dtype=np.uint8)
        success, encoded = cv2.imencode('.png', frame)
        self.assertTrue(success)
        content = encoded.tobytes()
        response = self.client.post('/api/frames', content=content, headers={'Content-Type':'image/png'})
        self.assertEqual(response.status_code, 201)
        result = response.json()
        self.assertEqual(result['sha256'], hashlib.sha256(content).hexdigest())
        self.assertTrue(result['quality']['passed'])
        self.assertEqual(self.client.get(result['image_url']).content, content)
        with patch.object(main.proxy_detector, 'path', None):
            response = self.client.post(f"/api/proxy/frames/{result['id']}/detect")
            self.assertEqual(response.status_code, 503)
        uniform = cv2.imencode('.png', np.full((128,128,3),120,dtype=np.uint8))[1].tobytes()
        rejected = self.client.post('/api/frames',content=uniform).json()
        self.assertEqual(self.client.post(f"/api/proxy/frames/{rejected['id']}/detect").status_code, 422)

    def test_concurrent_decisions_have_one_winner(self):
        record = self.replay()
        def submit(_):
            return self.client.post(f"/api/inspections/{record['id']}/decision", json={'decision':'approve','engineer':'Engineer A'}).status_code
        with ThreadPoolExecutor(max_workers=2) as executor: responses = list(executor.map(submit, range(2)))
        self.assertEqual(sorted(responses), [200,409])
        self.assertEqual(main.repo.verify_audit(record['id'])['events'], 2)

    def test_immutable_evidence_and_tamper_detection(self):
        record = self.replay()
        def mutate(payload):
            payload['telemetry']['temperature_c'] = 0
            return payload
        with self.assertRaises(ValueError): main.repo.update(record['id'], mutate)
        self.assertTrue(main.repo.verify_audit(record['id'])['valid'])
        from sqlalchemy.orm import Session
        with Session(main.repo.engine) as session, session.begin():
            row = session.get(Inspection, record['id'])
            payload = dict(row.payload); payload['telemetry'] = {'temperature_c':0}; row.payload = payload
        self.assertFalse(main.repo.verify_audit(record['id'])['valid'])

    def test_filters_auth_and_grounded_briefing(self):
        self.replay(); self.replay(scenario='normal')
        self.assertEqual(len(self.client.get('/api/inspections?lot_id=B127&limit=1').json()), 1)
        self.assertEqual(self.client.get('/api/inspections?machine_id=missing').json(), [])
        self.assertEqual(self.client.get('/api/inspections?limit=101').status_code, 422)
        record = self.replay()
        packet = self.client.get(f"/api/inspections/{record['id']}/briefing").json()
        self.assertFalse(packet['llm_used'])
        self.assertEqual(packet['facts']['source'], 'synthetic_replay')
        with patch.dict(os.environ, {'LINEGUARD_API_TOKEN':'test-secret'}):
            self.assertEqual(self.client.get('/api/inspections').status_code, 401)
            self.assertEqual(self.client.get('/api/health').status_code, 200)
            self.assertEqual(self.client.get('/api/inspections',headers={'Authorization':'Bearer test-secret'}).status_code, 200)

if __name__ == '__main__': unittest.main()
