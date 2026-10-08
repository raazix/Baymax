"""Register a validation-calibrated casting artifact without replacing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def activate(report_path):
    report_path = Path(report_path).resolve()
    weights = report_path.parent / 'casting_proxy.pt'
    report = json.loads(report_path.read_text())
    calibration = report['calibration']
    if not report_path.is_relative_to(ROOT) or not weights.is_relative_to(ROOT):
        raise ValueError('Registered evidence must remain in workspace')
    if calibration['selection_split'] != 'validation' or calibration['test_used_for_selection'] is not False:
        raise ValueError('Expected validation-only calibration')
    if report['brake_disc_validated'] is not False or report['production_validated'] is not False:
        raise ValueError('Proxy limitations must remain explicit')
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != report['artifact_sha256']:
        raise ValueError('Artifact differs from calibrated evaluation')
    from vision.patchcore_anomaly import PatchCoreDetector
    detector = PatchCoreDetector(weights)
    detector._load()
    if detector.threshold != calibration['threshold']:
        raise ValueError('Runtime threshold differs from calibration')
    registry = {'weights':weights.relative_to(ROOT).as_posix(), 'weights_sha256':digest,
                'evaluation':report_path.relative_to(ROOT).as_posix(), 'device':'0',
                'scope':'casting proxy; validation-selected low false-alarm operating point',
                'target_validation_normal_fpr':calibration['target_validation_normal_fpr'],
                'brake_disc_validated':False, 'production_validated':False}
    destination = ROOT / 'models/patchcore/active.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps(registry,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,default=ROOT/'models/patchcore/calibrated_low_fpr/evaluation.json')
    activate(parser.parse_args().report)
