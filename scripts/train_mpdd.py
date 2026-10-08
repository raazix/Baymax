"""Independent normal-only MPDD component PatchCore banks; frozen test evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
MANIFEST = ROOT / 'data/processed/mpdd/manifest.json'
CATEGORIES = ('bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes')


def normal_threshold(scores, target_fpr=.05):
    import numpy as np
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Expected nonempty finite nonnegative validation-normal distances')
    if isinstance(target_fpr, bool) or not math.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError('Invalid normal false-alarm budget')
    allowed = math.floor(len(scores) * target_fpr)
    threshold = float(np.sort(scores)[len(scores) - allowed - 1]) if allowed < len(scores) else 0.
    return max(threshold, float(np.nextafter(0., 1.)))


def validate_manifest(records):
    identities, paths = {}, set()
    for row in records:
        if row['path'] in paths:
            raise ValueError('Repeated image path in MPDD manifest')
        paths.add(row['path'])
        if row['category'] not in CATEGORIES or row['split'] not in ('train', 'validation', 'test') or row['label'] not in ('normal', 'defect'):
            raise ValueError('Unknown MPDD category, split or label')
        if row['split'] in ('train', 'validation') and (row['label'] != 'normal' or row['source_split'] != 'train'):
            raise ValueError('Training and validation must contain only official-training normals')
        if row['split'] == 'test' and row['source_split'] != 'test':
            raise ValueError('Test must contain only official-test images')
        signature = (row['category'], row['split'], row['label'])
        if row['sha256'] in identities and identities[row['sha256']] != signature:
            raise ValueError('Identical bytes cross category/split/label boundaries')
        identities[row['sha256']] = signature
    for category in {row['category'] for row in records}:
        for split in ('train', 'validation', 'test'):
            if not any(row['category'] == category and row['split'] == split for row in records):
                raise ValueError(f'Missing {category}/{split}')


def load_image(record):
    import cv2
    path = (ROOT / record['path']).resolve()
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Image path/hash differs from MPDD manifest')
    if record.get('mask_path'):
        mask_path = (ROOT / record['mask_path']).resolve()
        if not mask_path.is_relative_to(ROOT) or hashlib.sha256(mask_path.read_bytes()).hexdigest() != record['mask_sha256']:
            raise ValueError('Ground-truth mask path/hash differs from MPDD manifest')
    frame = cv2.imread(str(path))
    if frame is None or frame.shape[:2] != (record['height'], record['width']):
        raise ValueError('Cannot decode expected MPDD image dimensions')
    return frame


def test_metrics(records, scores, threshold):
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
    labels = np.array([row['label'] == 'defect' for row in records], dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not np.isfinite(scores).all() or len(set(labels)) != 2:
        raise ValueError('Test must have finite scores and both labels')
    predicted = scores > threshold
    tn = int(((labels == 0) & ~predicted).sum()); fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum()); tp = int(((labels == 1) & predicted).sum())
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary', zero_division=0)
    return {'auroc': float(roc_auc_score(labels, scores)), 'average_precision': float(average_precision_score(labels, scores)),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
            'confusion_matrix': [[tn, fp], [fn, tp]], 'normal_fpr': fp / (tn + fp),
            'false_positives': fp, 'normal_images': tn + fp, 'defect_images': tp + fn,
            'confusion_matrix_order': 'rows actual normal/defect, columns predicted normal/defect'}


def train_category(category, records, extractor, backbone, backbone_digest, device, batch_size, output_root, manifest_hash, seed):
    import numpy as np
    import torch
    from vision.patchcore_anomaly import extract_patch_features, greedy_coreset, nearest_patch_distances, square_crops
    began = time.time()
    folder = output_root / category
    if folder.exists():
        raise FileExistsError(f'Preserve prior category artifacts: {folder}')
    selected = [row for row in records if row['category'] == category]
    training = [row for row in selected if row['split'] == 'train']
    features = []
    training_crops = 0
    for index, record in enumerate(training):
        crops = square_crops(load_image(record))
        training_crops += len(crops)
        for begin in range(0, len(crops), batch_size):
            features.append(extract_patch_features(extractor, crops[begin:begin + batch_size]).flatten(0, 1).cpu())
        if index % 30 == 0:
            print(f'{category}: train features {index + 1}/{len(training)}', flush=True)
    bank, coreset = greedy_coreset(torch.cat(features).to(device), max_points=1024, seed=seed)
    del features
    def score(rows):
        scores = []
        for record in rows:
            crops = square_crops(load_image(record))
            maximum = 0.
            for begin in range(0, len(crops), batch_size):
                patches = extract_patch_features(extractor, crops[begin:begin + batch_size]).flatten(0, 1)
                maximum = max(maximum, float(nearest_patch_distances(patches, bank).max()))
            scores.append(maximum)
        return scores
    validation = [row for row in selected if row['split'] == 'validation']
    validation_scores = score(validation)
    threshold = normal_threshold(validation_scores)
    false_alarms = int((np.asarray(validation_scores) > threshold).sum())
    calibration = {'split': 'validation_normal_only', 'target_normal_fpr': .05,
                   'selection': 'Lowest positive threshold with floor(N*.05) allowed validation-normal alarms; strict score > threshold.',
                   'normal_images': len(validation), 'false_positives': false_alarms,
                   'observed_normal_fpr': false_alarms / len(validation), 'test_used_for_selection': False,
                   'population_fpr_guaranteed': False, 'production_approved': False}
    metadata = {'method': 'compact_patchcore_resnet18', 'data_source': 'mpdd_proxy', 'category': category,
                'model_version': 'mpdd-normal-patchcore-v1', 'brake_disc_validated': False, 'production_validated': False,
                'seed': seed, 'manifest_sha256': manifest_hash, 'training_normal_images': len(training),
                'training_defect_images': 0, 'training_crops': training_crops, 'coreset': coreset,
                'threshold': threshold, 'threshold_selection': calibration,
                'feature_pipeline': 'Shared frozen ImageNet ResNet18 pooled layer2/layer3; square_crops for nonsquare images; uncropped resize224 per crop; image score max patch distance across crops.',
                'limitations': 'MPDD component proxy; separate bank per category; no brake-disc/production validation. Small normal calibration samples cannot guarantee population FPR. Near-duplicate/physical-part leakage not excluded.',
                'pixel_metrics': 'Not evaluated; genuine masks retained and hashed, but no pixel performance or physical metrology claimed.'}
    folder.mkdir(parents=True)
    artifact_path = folder / 'casting_proxy.pt'
    torch.save({'format_version': 1, 'bank': bank.cpu().float(), 'threshold': threshold,
                'backbone_path': backbone.relative_to(ROOT).as_posix(), 'backbone_sha256': backbone_digest,
                'metadata': metadata}, artifact_path)
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    # Official test images are first loaded only AFTER the bank/threshold are frozen.
    test = [row for row in selected if row['split'] == 'test']
    test_scores = score(test)
    report = {'category': category, 'artifact_sha256': artifact_hash, 'data_source': 'mpdd_proxy',
              'brake_disc_validated': False, 'metadata': metadata, 'test': test_metrics(test, test_scores, threshold),
              'elapsed_seconds': time.time() - began,
              'scores': [{**row, 'anomaly_score': float(value), 'predicted_defective': value > threshold}
                         for rows, values in ((validation, validation_scores), (test, test_scores)) for row, value in zip(rows, values)]}
    (folder / 'metadata.json').write_text(json.dumps({**metadata, 'artifact_sha256': artifact_hash}, indent=2, allow_nan=False))
    (folder / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'category': category, 'threshold': threshold, 'validation': calibration, 'test': report['test'], 'artifact_sha256': artifact_hash}, indent=2), flush=True)
    return report


def train(categories=CATEGORIES, device='cuda:0', batch_size=8, output_root=ROOT / 'models/patchcore/mpdd', seed=42):
    import torch
    from scripts.download_patchcore_backbone import DESTINATION
    from vision.patchcore_anomaly import load_feature_extractor
    if not categories or any(category not in CATEGORIES for category in categories) or len(set(categories)) != len(categories):
        raise ValueError('Select unique known MPDD categories')
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    output_root = Path(output_root).resolve()
    if not output_root.is_relative_to(ROOT) or any((output_root / category).exists() for category in categories):
        raise ValueError('Use fresh workspace category output folders')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly choose --device cpu')
    records = json.loads(MANIFEST.read_text())['records']
    validate_manifest(records)
    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    if not digest.startswith('f37072fd'):
        raise ValueError('Official verified ImageNet backbone required')
    torch.set_num_threads(4); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    extractor = load_feature_extractor(DESTINATION, device)
    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    return [train_category(category, records, extractor, DESTINATION, digest, device, batch_size,
                           output_root, manifest_hash, seed) for category in categories]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', default=list(CATEGORIES), choices=CATEGORIES)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'models/patchcore/mpdd')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.categories, args.device, args.batch_size, args.output_root, args.seed)
