"""Exercise the live API and separate durable worker with synthetic evidence."""
import json
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]

def main():
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=30) as client:
        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()
        def post(path, body):
            response = client.post(path, json=body)
            response.raise_for_status()
            return response.json()
        models = get('/api/models')
        assert models['analytics_pipeline']['mode'] == 'trained-synthetic-v1'
        for name in ('xgboost_rca', 'plsr', 'pcr'):
            assert models[name]['configured'] and not models[name]['production_validated']
        inline = post('/api/replay', {'scenario': 'thermal_drift', 'seed': 42})
        deferred = post('/api/replay', {'scenario': 'thermal_drift', 'seed': 42, 'analytics_mode': 'deferred'})
        job_id = deferred['analytics_job_id']
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            job = get('/api/jobs/' + job_id)
            if job['status'] in ('completed', 'failed'):
                break
            time.sleep(.2)
        assert job['status'] == 'completed', job
        completed = get('/api/inspections/' + deferred['id'])
        assert completed['analytics'] == inline['analytics']
        assert completed['action']['containment'] == 'affected_lot'
        analytics = completed['analytics']
        assert analytics['rca']['method'] == 'XGBoost + exact TreeSHAP'
        assert not analytics['rca']['confidence_calibrated']
        assert abs(analytics['rca']['additivity_residual']) < 1e-4
        assert analytics['forecast']['pcr_role'] == 'benchmark only'
        assert analytics['uncertainty']['simulations'] == 10_000
        assert not analytics['production_validated']
        post('/api/inspections/' + deferred['id'] + '/decision', {'decision': 'approve', 'engineer': 'Synthetic API test engineer'})
        evidence = [post('/api/replay', {'scenario': 'normal'})['id'] for _ in range(20)]
        verified = post('/api/inspections/' + deferred['id'] + '/verification', {'inspection_ids': evidence})
        assert verified['verification']['defective'] == 0
        assert not verified['verification']['production_fix_verified']
        assert get('/api/inspections/' + deferred['id'] + '/audit')['integrity']['valid']
        briefing = get('/api/inspections/' + deferred['id'] + '/briefing')
        assert not briefing['llm_used'] and briefing['facts']['source'] == 'synthetic_replay'
        report = {'status': 'passed', 'source': 'synthetic_live_api_test', 'inspection_id': deferred['id'],
                  'job_id': job_id, 'model_versions': completed['model_versions'],
                  'rca_method': analytics['rca']['method'], 'forecast_method': analytics['forecast']['method'],
                  'audit_verified': True, 'synthetic_verification': verified['verification'],
                  'production_validated': False}
        destination = ROOT / 'data/analytics_api_test/report.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
