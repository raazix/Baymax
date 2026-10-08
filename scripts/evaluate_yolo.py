"""Evaluate saved YOLO weights without retraining, then persist provenance."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def evaluate(weights: Path, data: Path, device='0', batch=8, workers=0, elapsed_seconds=None,
             report_name='evaluation.json', reused_test=False):
    (ROOT / 'data/yolo-config/Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'data/yolo-config'))
    import torch
    import ultralytics
    from ultralytics import YOLO
    run = weights.resolve().parents[1]
    if Path(report_name).name != report_name or not report_name.endswith('.json'):
        raise ValueError('Report name must be a JSON filename')
    if (run / report_name).exists():
        raise ValueError('Evaluation already exists. Preserve the held-out report; do not repeatedly tune on the test set.')
    started = time.time()
    model = YOLO(str(weights))
    metrics = model.val(data=str(data.resolve()), split='test', imgsz=640, batch=batch,
                        device=device, workers=workers, project=str(ROOT / 'models/yolo'),
                        name=run.name + '_test', plots=True)
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, map50, map5095 = metrics.box.class_result(index)
        per_class.append({'class': model.names[int(class_id)], 'precision': float(precision),
                          'recall': float(recall), 'mAP50': float(map50), 'mAP50_95': float(map5095)})
    manifest = data.parent / 'manifest.json'
    rows = list(csv.DictReader((run / 'results.csv').open()))
    report = {'task': model.task, 'dataset': str(data.resolve()), 'evaluation_split': 'test',
              'metrics': {str(k):float(v) for k,v in metrics.results_dict.items()}, 'per_class': per_class,
              'weights': str(weights.resolve()), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'dataset_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'seed': 42, 'completed_epochs': int(float(rows[-1]['epoch'])),
              'evaluation_elapsed_seconds': round(time.time()-started,2),
              'training_process_elapsed_seconds': elapsed_seconds,
              'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
              'gpu': torch.cuda.get_device_name(int(device)) if device.isdigit() else device,
              'test_inference_speed_ms_per_image': metrics.speed,
              'brake_disc_validated': False, 'scope': 'NEU steel-surface detection proxy',
              'test_previously_used_for_baseline_reporting': reused_test,
              'fresh_blind_test': not reused_test,
              'used_for_candidate_selection': False,
              'limitation': 'Image/hash-group holdout; physical part identities unavailable; not brake-disc performance.'}
    schedule_path = run / 'training_schedule.json'
    if schedule_path.exists(): report['training_schedule'] = json.loads(schedule_path.read_text())
    (run / report_name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/yolo/neu_yolo11n/weights/best.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/neu-det/dataset.yaml')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--report-name', default='evaluation.json')
    parser.add_argument('--reused-test', action='store_true')
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.device, args.batch, args.workers,
             report_name=args.report_name, reused_test=args.reused_test)
