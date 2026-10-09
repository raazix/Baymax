"""Domain thresholds for the corrosion classifier.

The classifier's own threshold was chosen on general rust photos. On MPDD metal plates (colour photos of industrial
parts) it flags almost every plate, so a plate-specific threshold is set from NORMAL training plates only: the maximum
corrosion score over MPDD metal_plate train/good (the same low-false-alarm rule used for PatchCore). The MPDD test split
(good / scratches / major_rust / total_rust) is then scored once. Output: models/rust/domain_thresholds.json.
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config  # noqa: E402,F401
from vision.rust_classifier import RustClassifier  # noqa: E402

MPDD = ROOT / 'data/raw/MPDD/metal_plate'


def scores(classifier, folder):
    files = sorted(glob.glob(str(folder / '*.png')))
    return files, np.array([classifier.detect(cv2.imread(f), domain=None)['corrosion_probability'] for f in files])


def main():
    classifier = RustClassifier()
    train_files, train_good = scores(classifier, MPDD / 'train/good')
    threshold = float(train_good.max())
    test = {kind: scores(classifier, MPDD / 'test' / kind)[1] for kind in ('good', 'scratches', 'major_rust', 'total_rust')}
    rust = np.r_[test['major_rust'], test['total_rust']]
    other = np.r_[test['good'], test['scratches']]
    report = {
        'mpdd_metal_plate': {
            'corrosion': round(threshold, 4), 'rule': f'maximum corrosion score over {len(train_files)} MPDD metal_plate train/good images',
            'train_good_sha256': hashlib.sha256(''.join(sorted(Path(f).name for f in train_files)).encode()).hexdigest(),
            'test': {'rust_vs_other_auroc': round(float(roc_auc_score(np.r_[np.ones(len(rust)), np.zeros(len(other))], np.r_[rust, other])), 4),
                     'rust_recall': f'{int((rust > threshold).sum())}/{len(rust)}',
                     'good_false_alarms': f'{int((test["good"] > threshold).sum())}/{len(test["good"])}',
                     'scratches_flagged': f'{int((test["scratches"] > threshold).sum())}/{len(test["scratches"])}'},
            'note': 'Presence only: the severe head is not recalibrated for plates, so severe grades on plates are not reported.'},
        'artifact_sha256': classifier.digest,
    }
    out = ROOT / 'models/rust/domain_thresholds.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
