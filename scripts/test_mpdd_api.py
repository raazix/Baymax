"""Functional checks for all six component-specific MPDD API models."""
import json
from pathlib import Path
import cv2
import httpx
from vision.quality_gate import check_frame_open_domain

ROOT = Path(__file__).resolve().parents[1]

def main():
    results = []
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=60) as client:
        catalog = client.get('/api/patchcore/models');catalog.raise_for_status()
        assert len(catalog.json()['mpdd']) == 6
        for entry in catalog.json()['mpdd']:
            category = entry['category']
            report = json.loads((ROOT/'models/patchcore/mpdd'/category/'evaluation.json').read_text())
            sample = next(r for r in sorted(report['scores'],key=lambda r:r['path']) if r['split']=='test'
                          and check_frame_open_domain(cv2.imread(str(ROOT/r['path'])),'camera')['passed'])
            response = client.post('/api/frames',content=(ROOT/sample['path']).read_bytes(),headers={'Content-Type':'image/png'})
            response.raise_for_status();frame=response.json()
            response = client.post('/api/proxy/frames/'+frame['id']+'/anomaly',params={'patchcore_model':entry['name']})
            response.raise_for_status();run=response.json()
            assert run['artifact_sha256'] == report['artifact_sha256']
            assert run['source'] == 'trained_mpdd_patchcore_proxy'
            assert run['provenance']['category'] == category
            assert run['image_sha256'] == sample['sha256']
            assert abs(run['anomaly_score']-sample['anomaly_score']) < .001
            saved=client.get('/api/proxy/runs/'+run['id']);saved.raise_for_status()
            assert saved.json()==run
            results.append({'category':category,'model':entry['name'],'run_id':run['id'],
                            'artifact_sha256':run['artifact_sha256'],'sample':sample['path'],'status':'passed'})
        assert client.post('/api/proxy/frames/missing/anomaly',params={'patchcore_model':'mpdd_invalid'}).status_code == 422
    output=ROOT/'data/mpdd_api_test/report.json';output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise ValueError('Preserve existing functional report')
    output.write_text(json.dumps({'status':'passed','scope':'first quality-passed test image per component; functional checks only',
                                  'models':results,'brake_disc_validated':False},indent=2))
    print('All six MPDD model API checks passed')

if __name__ == '__main__': main()
