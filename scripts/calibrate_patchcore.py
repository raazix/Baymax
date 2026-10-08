"""Freeze a validation-only low-false-alarm threshold; preserve the trained bank."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'models/patchcore/calibrated_low_fpr'


def validate_scores(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError('Labels and scores must be nonempty matching vectors')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Expected binary labels and finite nonnegative distances')
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError('Both normal and defective examples are required')
    return labels.astype(int), scores


def choose_low_fpr_threshold(labels, scores, target_fpr=.05):
    """Lowest positive strict-> threshold satisfying validation-normal FPR budget."""
    labels, scores = validate_scores(labels, scores)
    if isinstance(target_fpr, bool) or not math.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError('FPR budget must be a finite fraction in [0,1]')
    normal = np.sort(scores[labels == 0])
    allowed = math.floor(target_fpr * len(normal))
    threshold = float(normal[len(normal) - allowed - 1]) if allowed < len(normal) else 0.
    threshold = max(threshold, float(np.nextafter(0., 1.)))
    return threshold


def metrics(labels, scores, threshold):
    labels, scores = validate_scores(labels, scores)
    predicted = scores > threshold
    tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    tp = int(((labels == 1) & predicted).sum())
    return {'normal_images': tn + fp, 'defect_images': fn + tp, 'false_positives': fp,
            'false_negatives': fn, 'fpr': fp / (tn + fp), 'recall': tp / (tp + fn),
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'confusion_matrix': [[tn, fp], [fn, tp]],
            'order': 'rows actual normal/defect, columns predicted normal/defect'}


def vectors(records):
    return [int(row['label'] == 'defect') for row in records], [row['anomaly_score'] for row in records]


def calibrate(output=DEFAULT_OUTPUT, target_fpr=.05):
    import torch
    original = ROOT / 'models/patchcore/casting_proxy.pt'
    evaluation_path = ROOT / 'models/patchcore/evaluation.json'
    manifest_path = ROOT / 'data/processed/casting/manifest.json'
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    report = json.loads(evaluation_path.read_text())
    if report['artifact_sha256'] != original_hash:
        raise ValueError('Original artifact changed since scores were computed')
    artifact = torch.load(original, map_location='cpu', weights_only=True)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if artifact['metadata']['manifest_sha256'] != manifest_hash or report['metadata']['manifest_sha256'] != manifest_hash:
        raise ValueError('Casting manifest differs from original score provenance')
    expected = {row['path']: row for row in json.loads(manifest_path.read_text())['records'] if row['split'] in ('validation', 'test')}
    seen = set()
    for row in report['scores']:
        if row['path'] in seen or row['path'] not in expected:
            raise ValueError('Repeated or unexpected score record')
        seen.add(row['path'])
        if any(row[key] != expected[row['path']][key] for key in ('sha256', 'label', 'split')):
            raise ValueError('Score identity conflicts with frozen manifest')
    if seen != set(expected):
        raise ValueError('Missing frozen evaluation records')
    validation = [row for row in report['scores'] if row['split'] == 'validation']
    labels, scores = vectors(validation)
    threshold = choose_low_fpr_threshold(labels, scores, target_fpr)
    selected_metrics = metrics(labels, scores, threshold)
    calibration = {'model_version': 'casting-patchcore-validation-low-fpr-v1',
                   'objective_predeclared': 'Lowest positive strict score>threshold meeting validation-normal FPR budget; maximizes recall under this budget.',
                   'target_validation_normal_fpr': target_fpr, 'engineering_demo_target_only': True,
                   'production_approved': False, 'threshold': threshold,
                   'selection_split': 'validation', 'test_used_for_selection': False,
                   'validation': selected_metrics, 'original_validation': metrics(labels, scores, artifact['threshold']),
                   'source_artifact_sha256': original_hash, 'source_evaluation_sha256': hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
                   'source_manifest_sha256': manifest_hash,
                   'bank_tensor_sha256': hashlib.sha256(artifact['bank'].contiguous().numpy().tobytes()).hexdigest(),
                   'validation_records_sha256': hashlib.sha256(json.dumps(validation, sort_keys=True).encode()).hexdigest(),
                   'limitation': 'Observed validation FPR is not a population guarantee; existing test is a reused benchmark, not a new blind holdout.'}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'casting_proxy.pt'
    if destination.exists():
        raise ValueError('Versioned artifact already exists; choose a new output directory')
    derived = {**artifact, 'threshold': threshold,
               'metadata': {**artifact['metadata'], 'threshold': threshold, 'threshold_selection': calibration,
                            'model_version': calibration['model_version']}}
    # Freeze derived bank and threshold BEFORE evaluating reused test records.
    torch.save(derived, destination)
    frozen_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    (output / 'metadata.json').write_text(json.dumps(derived['metadata'], indent=2))
    test = [row for row in report['scores'] if row['split'] == 'test']
    test_labels, test_scores = vectors(test)
    result = {'artifact_sha256': frozen_hash, 'calibration': calibration,
              'test': metrics(test_labels, test_scores, threshold),
              'original_test': metrics(test_labels, test_scores, artifact['threshold']),
              'test_status': 'Reused benchmark already inspected before calibration; not a new blind holdout.',
              'test_used_for_selection': False, 'production_validated': False, 'brake_disc_validated': False}
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-fpr', type=float, default=.05)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    calibrate(args.output, args.target_fpr)
