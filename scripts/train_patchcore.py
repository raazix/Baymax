"""Normal-only casting memory bank, validation threshold, frozen final test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from scripts.prepare_casting import ROOT, MANIFEST
from scripts.download_patchcore_backbone import DESTINATION
from vision.patchcore_anomaly import load_feature_extractor, extract_patch_features, nearest_patch_distances, greedy_coreset


def load_images(records):
    images = []
    for record in records:
        path = ROOT / record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Image changed since split preparation: {path}')
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot decode {path}')
        images.append(frame)
    return images


def choose_threshold(labels, scores):
    # Runtime decision uses score > threshold; include positive minimum sentinel.
    candidates = np.unique(np.r_[np.nextafter(min(scores), 0), scores])
    best = None
    for threshold in candidates:
        if threshold <= 0:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(labels, np.asarray(scores) > threshold, average='binary', zero_division=0)
        candidate = (float(f1), float(precision), float(threshold), float(recall))
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise ValueError('No positive threshold candidate')
    return best[2], {'f1': best[0], 'precision': best[1], 'recall': best[3],
                     'selection': 'Maximum validation F1; ties prefer precision, then larger threshold. Prediction is score > threshold.'}


def metrics(records, scores, threshold):
    labels = np.array([record['label'] == 'defect' for record in records], dtype=int)
    prediction = np.asarray(scores) > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average='binary', zero_division=0)
    return {'auroc': float(roc_auc_score(labels, scores)), 'average_precision': float(average_precision_score(labels, scores)),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
            'confusion_matrix': confusion_matrix(labels, prediction, labels=[0,1]).tolist(),
            'confusion_matrix_order': 'rows actual normal/defect, columns predicted normal/defect',
            'images': len(records)}


def train(device='cuda:0', batch_size=8, seed=42, output_dir=None):
    began = time.time()
    folder = Path(output_dir or ROOT / 'models/patchcore').resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError('Keep model artifacts within the workspace')
    if any((folder / name).exists() for name in ('casting_proxy.pt', 'evaluation.json', 'metadata.json')):
        raise FileExistsError('Preserve trained artifacts and reports; choose a new --output-dir for another experiment.')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly select --device cpu for CPU training')
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    manifest = json.loads(MANIFEST.read_text())
    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    if not digest.startswith('f37072fd'):
        raise ValueError('Verified official pretrained backbone is required; refusing random/untrusted weights')
    extractor = load_feature_extractor(DESTINATION, device)
    records = manifest['records']
    training = [r for r in records if r['split'] == 'train']
    if not training or any(r['label'] != 'normal' for r in training):
        raise ValueError('Memory bank training must contain only normal images')
    features = []
    for begin in range(0, len(training), batch_size):
        features.append(extract_patch_features(extractor, load_images(training[begin:begin+batch_size])).flatten(0,1).cpu())
        if begin % (batch_size*10) == 0:
            print(f'Normal features {min(begin+batch_size,len(training))}/{len(training)}', flush=True)
    bank, coreset = greedy_coreset(torch.cat(features).to(device), seed=seed)
    del features
    print(f'Frozen normal bank: {tuple(bank.shape)}', flush=True)
    def score(split):
        selected = [r for r in records if r['split'] == split]
        scores = []
        for begin in range(0, len(selected), batch_size):
            patches = extract_patch_features(extractor, load_images(selected[begin:begin+batch_size]))
            distances = nearest_patch_distances(patches.flatten(0,1), bank)
            scores.extend(map(float, distances.reshape(len(patches), -1).amax(1).cpu()))
            if begin % (batch_size*10) == 0:
                print(f'{split} scores {min(begin+batch_size,len(selected))}/{len(selected)}', flush=True)
        return selected, scores
    validation, validation_scores = score('validation')
    threshold, selection = choose_threshold([r['label'] == 'defect' for r in validation], validation_scores)
    metadata = {'method': 'compact_patchcore_resnet18', 'data_source': 'casting_proxy', 'brake_disc_validated': False,
                'seed': seed, 'training_normal_images': len(training), 'training_defect_images': 0,
                'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), 'split_counts': manifest['counts'],
                'physical_part_ids_available': False, 'near_duplicate_check_performed': False,
                'coreset': coreset, 'threshold_selection': selection, 'threshold': threshold,
                'image_score': 'Maximum nearest normal-patch Euclidean distance',
                'feature_pipeline': 'Frozen official ImageNet ResNet18 layer2+layer3 pooled features, 224x224 uncropped RGB normalization; shared runtime extractor',
                'limitations': 'Casting proxy only; no pixel masks or defect-class-specific unknown anomaly validation. Scores are not probabilities; physical-part split leakage cannot be excluded.'}
    folder.mkdir(parents=True, exist_ok=True)
    artifact_path = folder / 'casting_proxy.pt'
    artifact = {'format_version': 1, 'bank': bank.cpu().float(), 'threshold': threshold,
                'backbone_path': DESTINATION.relative_to(ROOT).as_posix(), 'backbone_sha256': digest, 'metadata': metadata}
    torch.save(artifact, artifact_path)
    artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    # The artifact and threshold are frozen before the test images are loaded.
    test, test_scores = score('test')
    report = {'data_source': 'casting_proxy', 'brake_disc_validated': False, 'artifact_sha256': artifact_digest,
              'metadata': metadata, 'validation': metrics(validation, validation_scores, threshold),
              'test': metrics(test, test_scores, threshold), 'elapsed_seconds': time.time()-began,
              'scores': [{**record, 'anomaly_score': score_value, 'predicted_defective': score_value > threshold}
                         for split_records, split_scores in ((validation,validation_scores),(test,test_scores))
                         for record, score_value in zip(split_records,split_scores)]}
    (folder / 'metadata.json').write_text(json.dumps({**metadata, 'artifact_sha256': artifact_digest}, indent=2, allow_nan=False))
    (folder / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: report[key] for key in ('artifact_sha256','validation','test','elapsed_seconds')}, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models/patchcore')
    args = parser.parse_args()
    train(args.device, args.batch_size, args.seed, args.output_dir)
