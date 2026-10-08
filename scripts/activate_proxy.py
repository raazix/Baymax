"""Register an evaluated NEU model for backend startup without shell env setup."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def activate(report_path: Path):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('evaluation_split') != 'test' or report.get('brake_disc_validated') is not False:
        raise ValueError('Expected a held-out evaluation with explicit proxy limitations')
    weights = Path(report['weights']).resolve()
    if not weights.is_relative_to(ROOT): raise ValueError('Weights must be in the project workspace')
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != report['weights_sha256']: raise ValueError('Weights differ from the evaluated model')
    (ROOT / 'data/yolo-config/Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'data/yolo-config'))
    from ultralytics import YOLO
    from prepare_neu import CLASSES
    model = YOLO(str(weights))
    if model.task != 'detect' or list(model.names.values()) != CLASSES:
        raise ValueError('Expected the six original NEU detection labels')
    registry = {'weights': weights.relative_to(ROOT).as_posix(), 'weights_sha256': digest,
                'evaluation': report_path.resolve().relative_to(ROOT).as_posix(),
                'device': '0', 'scope': 'NEU steel-surface proxy', 'brake_disc_validated': False}
    destination = ROOT / 'models' / 'yolo' / 'active.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(registry, indent=2), encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps(registry, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=ROOT / 'models/yolo/neu_yolo11n/evaluation.json')
    activate(parser.parse_args().report)
