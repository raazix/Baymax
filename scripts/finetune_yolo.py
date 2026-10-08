"""Conservative proxy fine-tuning; select exclusively on validation, never test."""
import hashlib
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def evaluate(model, name):
    metrics = model.val(data=str(ROOT / 'data/processed/neu-det/dataset.yaml'), split='val',
                        imgsz=640, batch=8, device='0', workers=2,
                        project=str(ROOT / 'models/yolo'), name=name, plots=True)
    return {'metrics': {k: float(v) for k, v in metrics.results_dict.items()},
            'per_class': [{'class': model.names[int(c)], 'precision': float(metrics.box.p[i]),
                           'recall': float(metrics.box.r[i]), 'mAP50': float(metrics.box.ap50[i]),
                           'mAP50_95': float(metrics.box.ap[i].mean())}
                          for i, c in enumerate(metrics.box.ap_class_index)]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['conservative', 'rebalanced'], default='conservative')
    strategy = parser.parse_args().strategy
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'data/yolo-config'))
    from ultralytics import YOLO
    baseline = ROOT / 'models/yolo/neu_yolo11n/weights/best.pt'
    data = ROOT / 'data/processed/neu-det/dataset.yaml'
    if strategy == 'rebalanced':
        import yaml
        folder = ROOT / 'data/processed/neu-finetune'
        folder.mkdir(parents=True, exist_ok=True)
        images = sorted((data.parent / 'images/train').glob('*.jpg'))
        difficult = [p for p in images if p.stem.startswith('crazing_')]
        # Repeat only original training image paths; no copies cross any split.
        listing = folder / 'train.txt'
        listing.write_text('\n'.join(p.resolve().as_posix() for p in images + difficult * 2), encoding='utf-8')
        configuration = yaml.safe_load(data.read_text())
        configuration['train'] = listing.resolve().as_posix()
        data = folder / 'dataset.yaml'
        data.write_text(yaml.safe_dump(configuration), encoding='utf-8')
    # Declare the selection criterion before training; original test remains untouched.
    criterion = 'Candidate validation mAP50-95 and crazing AP50 must both exceed baseline.'
    baseline_metrics = evaluate(YOLO(str(baseline)), 'finetune_baseline_val')
    model = YOLO(str(baseline))
    model.train(data=str(data), epochs=12 if strategy == 'rebalanced' else 20,
                imgsz=640, batch=8, workers=2, device='0', optimizer='AdamW',
                lr0=0.0003 if strategy == 'rebalanced' else 0.0002, lrf=0.1, warmup_epochs=2,
                mosaic=0 if strategy == 'rebalanced' else 0.25, close_mosaic=5,
                seed=42, deterministic=True, patience=8, save_period=5, plots=True,
                project=str(ROOT / 'models/yolo'), name='neu_yolo11n_rebalanced' if strategy == 'rebalanced' else 'neu_yolo11n_finetuned')
    run = Path(model.trainer.save_dir)
    weights = run / 'weights/best.pt'
    candidate = evaluate(YOLO(str(weights)), run.name + '_val')
    def crazing(report):
        return next(c['mAP50'] for c in report['per_class'] if c['class'] == 'crazing')
    selected = (candidate['metrics']['metrics/mAP50-95(B)'] > baseline_metrics['metrics']['metrics/mAP50-95(B)']
                and crazing(candidate) > crazing(baseline_metrics))
    report = dict(candidate, baseline=baseline_metrics, selection_criterion=criterion,
                  selected=selected, evaluation_split='val', test_used_for_selection=False,
                  weights=str(weights.resolve()), weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  brake_disc_validated=False, scope='NEU steel-surface detection proxy',
                  strategy=strategy, dataset_manifest_sha256=hashlib.sha256((ROOT / 'data/processed/neu-det/manifest.json').read_bytes()).hexdigest(),
                  training_list_sha256=hashlib.sha256(listing.read_bytes()).hexdigest() if strategy == 'rebalanced' else None,
                  sampling_note='Crazing source training images repeated 3 times; validation unchanged.' if strategy == 'rebalanced' else 'Original training distribution.',
                  completed_epochs=model.trainer.epoch + 1)
    (run / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if selected:
        registry = dict(weights=weights.relative_to(ROOT).as_posix(), weights_sha256=report['weights_sha256'],
                        evaluation=(run / 'evaluation.json').relative_to(ROOT).as_posix(),
                        evaluation_split='val', device='0', scope=report['scope'], brake_disc_validated=False)
        destination = ROOT / 'models/yolo/active.json'
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(registry, indent=2), encoding='utf-8')
        temporary.replace(destination)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
