"""Train and evaluate a component model with reproducible provenance."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Ultralytics dataset YAML')
    parser.add_argument('--task', choices=['detect', 'segment'], default='segment')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=0, help='Zero avoids Windows multiprocessing issues')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--name', default='neu_yolo11n')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--resume', default=None, help='Path to an interrupted run last.pt')
    args = parser.parse_args()
    if not Path(args.data).is_file(): parser.error('Dataset YAML does not exist')
    if args.epochs < 1 or args.batch < 1: parser.error('Epochs and batch must be positive')
    root = Path(__file__).resolve().parents[1]
    (root / 'data' / 'yolo-config' / 'Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(root / 'data' / 'yolo-config'))
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        parser.error('SciPy import failed. Install scipy in the venv before training; it is needed for final diagnostic plots.')
    import torch
    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Install CUDA-enabled PyTorch and check the NVIDIA driver; choose --device cpu only explicitly.')
    from ultralytics import YOLO
    initial_weights = args.resume or args.weights or ('yolo11n-seg.pt' if args.task == 'segment' else 'yolo11n.pt')
    model = YOLO(initial_weights)
    started = time.time()
    model.train(data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=str(root / 'models' / 'yolo'), name=args.name,
                workers=args.workers, seed=42, deterministic=True, patience=10,
                save=True, save_period=5, plots=True, resume=bool(args.resume))
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / 'weights' / 'best.pt'
    if not best.is_file(): raise RuntimeError('Training finished without best.pt')
    # Test set is used once for final reporting, never to select an epoch.
    evaluated = YOLO(str(best))
    metrics = evaluated.val(data=str(Path(args.data).resolve()), split='test', imgsz=args.imgsz,
                            batch=args.batch, device=args.device, workers=args.workers,
                            project=str(root / 'models' / 'yolo'), name=run_dir.name + '_test', plots=True)
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, map50, map5095 = metrics.box.class_result(index)
        per_class.append({'class': evaluated.names[int(class_id)], 'precision': float(precision),
                          'recall': float(recall), 'mAP50': float(map50), 'mAP50_95': float(map5095)})
    manifest = Path(args.data).parent / 'manifest.json'
    report = {'task': args.task, 'dataset': str(Path(args.data).resolve()), 'evaluation_split': 'test',
              'metrics': {str(k):float(v) for k,v in metrics.results_dict.items()}, 'per_class': per_class,
              'weights': str(best.resolve()), 'weights_sha256': hashlib.sha256(best.read_bytes()).hexdigest(),
              'dataset_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
              'seed': 42, 'requested_epochs': args.epochs, 'completed_epochs': model.trainer.epoch+1,
              'elapsed_seconds': round(time.time()-started,2), 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(int(args.device)) if args.device.isdigit() else args.device,
              'brake_disc_validated': False, 'scope': 'NEU steel-surface detection proxy',
              'limitation': 'Image/hash-group holdout; physical part identities unavailable; not brake-disc performance.'}
    (run_dir / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('\nBackend configuration:')
    print(f"$env:LINEGUARD_PROXY_WEIGHTS = '{best.resolve().as_posix()}'")
    print("$env:LINEGUARD_MODEL_DEVICE = '0'")

if __name__ == '__main__': main()
