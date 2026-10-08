"""Frozen casting proxy through the live quality-gated anomaly API."""
import hashlib
import argparse
import json
from pathlib import Path
import time
import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/patchcore_api_test')
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'models/patchcore/evaluation.json')
    parser.add_argument('--score-reference', type=Path, default=ROOT / 'models/patchcore/evaluation.json')
    parser.add_argument('--quality-profile', choices=['camera', 'casting_proxy'], default='camera')
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    if 'scores' not in benchmark:
        reference_bytes = args.score_reference.read_bytes()
        assert hashlib.sha256(reference_bytes).hexdigest() == benchmark['calibration']['source_evaluation_sha256']
        reference = json.loads(reference_bytes)
        assert reference['artifact_sha256'] == benchmark['calibration']['source_artifact_sha256']
    else:
        reference = benchmark
    samples = sorted((r for r in reference['scores'] if r['split'] == 'test'), key=lambda r: r['path'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists(): raise ValueError('Preserve existing API test report.')
    results, durations, examples = [], [], {}
    tp = fp = tn = fn = rejected = 0
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=60) as client:
        def post(path, **kwargs):
            response = client.post(path, **kwargs)
            response.raise_for_status()
            return response.json()
        for index, sample in enumerate(samples):
            path = ROOT / sample['path']
            content = path.read_bytes()
            assert hashlib.sha256(content).hexdigest() == sample['sha256']
            frame = post('/api/frames?quality_profile='+args.quality_profile, content=content, headers={'Content-Type':'image/jpeg'})
            assert frame['sha256'] == sample['sha256']
            started = time.perf_counter()
            response = client.post('/api/proxy/frames/' + frame['id'] + '/anomaly')
            elapsed = (time.perf_counter() - started) * 1000
            if not frame['quality']['passed']:
                assert response.status_code == 422
                rejected += 1
                result = {'quality_passed':False}
            else:
                response.raise_for_status()
                result = response.json()
                assert result['artifact_sha256'] == benchmark['artifact_sha256']
                assert result['image_sha256'] == sample['sha256']
                assert result['threshold'] == benchmark.get('metadata', benchmark.get('calibration'))['threshold']
                assert abs(result['anomaly_score'] - sample['anomaly_score']) < .001
                assert result['metrology'] is None and not result['brake_disc_validated']
                grid = np.asarray(result['anomaly_grid'])
                assert grid.shape == (28,28) and np.isfinite(grid).all()
                assert abs(float(grid.max()) - result['anomaly_score']) < 1e-6
                saved = client.get('/api/proxy/runs/' + result['id']); saved.raise_for_status()
                assert saved.json() == result
                predicted = result['label'] == 'ANOMALY_UNCLASSIFIED'
                actual = sample['label'] == 'defect'
                assert predicted == (result['anomaly_score'] > result['threshold'])
                tp += int(predicted and actual); fp += int(predicted and not actual)
                tn += int(not predicted and not actual); fn += int(not predicted and actual)
                durations.append(elapsed)
                if sample['label'] not in examples:
                    examples[sample['label']] = (path, grid, result['anomaly_score'], result['label'])
                result['quality_passed'] = True
            results.append({'path':sample['path'], 'ground_truth':sample['label'], 'frame_id':frame['id'],
                            'quality_passed':result['quality_passed'], 'run_id':result.get('id'),
                            'anomaly_score':result.get('anomaly_score'), 'request_ms':elapsed})
            if (index+1) % 100 == 0: print(f'PatchCore API tested {index+1}/{len(samples)}', flush=True)
        assert client.post('/api/proxy/frames/missing/anomaly').status_code == 404
        blank = cv2.imencode('.png',np.full((64,64,3),100,dtype=np.uint8))[1].tobytes()
        blurred = post('/api/frames',content=blank,headers={'Content-Type':'image/png'})
        assert client.post('/api/proxy/frames/'+blurred['id']+'/anomaly').status_code == 422
        status = client.get('/api/models'); status.raise_for_status()
        assert status.json()['patchcore']['loaded']
        assert status.json()['patchcore']['model_sha256'] == benchmark['artifact_sha256']
    summary = {'tested_images':len(samples), 'quality_passed':len(durations), 'quality_rejected':rejected,
               'quality_profile':args.quality_profile,
               'artifact_sha256':benchmark['artifact_sha256'], 'persisted_results_verified':len(durations),
               'confusion_matrix_on_accepted_images':[[tn,fp],[fn,tp]],
               'precision_on_accepted_images':tp/(tp+fp) if tp+fp else None,
               'recall_on_accepted_images':tp/(tp+fn) if tp+fn else None,
               'normal_false_alarm_rate_on_accepted_images':fp/(tn+fp) if tn+fp else None,
               'median_inference_request_ms':float(np.median(durations)) if durations else None,
               'p95_inference_request_ms':float(np.quantile(durations,.95)) if durations else None,
               'scope':'Local inference HTTP request including decode and result DB commit; upload/quality request excluded.',
               'threshold_unchanged':True, 'data_source':'casting_proxy', 'brake_disc_validated':False}
    (output / 'report.json').write_text(json.dumps({'summary':summary,'images':results},indent=2,allow_nan=False))
    if examples:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(len(examples),2,figsize=(9,4*len(examples)),squeeze=False)
        for row,(label,(path,grid,score,prediction)) in enumerate(sorted(examples.items())):
            axes[row,0].imshow(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB))
            axes[row,0].set_title(f'{label} / {path.name}');axes[row,0].axis('off')
            plot = axes[row,1].imshow(grid,cmap='inferno',interpolation='nearest')
            axes[row,1].set_title(f'{prediction}: {score:.3f}\n28x28 patch distances, not segmentation')
            fig.colorbar(plot,ax=axes[row,1],label='Feature distance')
        fig.suptitle('Casting proxy: first quality-passed test image per label')
        fig.tight_layout();fig.savefig(output/'examples.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
