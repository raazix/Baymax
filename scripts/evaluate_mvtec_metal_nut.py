"""Evaluate the separately trained MVTec metal_nut PatchCore model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.patchcore_anomaly import PatchCoreDetector
DATA = ROOT / 'data/processed/mvtec_ad/metal_nut'
MODEL = ROOT / 'models/patchcore/custom/mvtec_metal_nut/casting_custom.pt'


def main() -> None:
    if not DATA.is_dir() or not MODEL.is_file():
        raise SystemExit('Run scripts/train_mvtec_metal_nut.py first.')
    detector = PatchCoreDetector(artifact_path=MODEL)
    image_scores: list[float] = []
    image_labels: list[int] = []
    pixel_scores: list[float] = []
    pixel_labels: list[int] = []
    classes: dict[str, dict] = {}
    for folder in sorted((DATA / 'test').iterdir()):
        if not folder.is_dir():
            continue
        label = folder.name
        tp = fp = fn = tn = 0
        paths = sorted(folder.glob('*.png'))
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Unable to decode {path}')
            result = detector.detect(frame)
            defective = label != 'good'
            predicted = result['anomaly_score'] > result['threshold']
            tp += int(defective and predicted)
            fn += int(defective and not predicted)
            fp += int(not defective and predicted)
            tn += int(not defective and not predicted)
            image_scores.append(float(result['anomaly_score']))
            image_labels.append(int(defective))

            score_map = cv2.resize(np.asarray(result['anomaly_grid'], np.float32), (64, 64), interpolation=cv2.INTER_CUBIC)
            pixel_scores.extend(score_map.reshape(-1).tolist())
            if defective:
                mask_path = DATA / 'ground_truth' / label / f'{path.stem}_mask.png'
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(mask_path)
                truth = cv2.resize((mask > 0).astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
            else:
                truth = np.zeros((64, 64), np.uint8)
            pixel_labels.extend(truth.reshape(-1).tolist())
        classes[label] = {
            'images': len(paths), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'defect_recall': tp / (tp + fn) if tp + fn else None,
        }

    confusion = {key: sum(row[key] for row in classes.values()) for key in ('tp', 'fp', 'fn', 'tn')}
    result = {
        'dataset': 'MVTec AD 2019 metal_nut',
        'license': 'CC BY-NC-SA 4.0',
        'threshold': detector.threshold,
        'threshold_source': '44 held-out normals from official training split; official test split untouched until evaluation',
        'test_image_auroc': roc_auc_score(image_labels, image_scores),
        'test_pixel_auroc_64x64_interpolated_patch_grid': roc_auc_score(pixel_labels, pixel_scores),
        'confusion_at_normal_calibration_threshold': confusion,
        'per_defect_class': classes,
        'artifact_sha256': detector.status()['artifact_sha256'],
        'brake_disc_validated': False,
        'pixel_metric_note': 'AUROC samples the 64x64 interpolation of PatchCore grid scores against resized binary masks; it is not mask AP or native-resolution segmentation accuracy.',
    }
    output = DATA / 'evaluation.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
