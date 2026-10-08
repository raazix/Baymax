"""Train a PatchCore memory bank for a new casting from the user's own NORMAL images.

PatchCore only needs examples of good parts, so this is how it is pointed at "any casting". Honest limits:
- The threshold comes from held-out normal images only (no defect examples exist), so the false-alarm rate is
  estimated from a small sample and detection performance on real defects of that part is unmeasured.
- Held-out images are chosen deterministically by content hash and never enter the memory bank.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vision.patchcore_anomaly import (ROOT, load_feature_extractor, extract_patch_features, nearest_patch_distances,
                                      greedy_coreset, square_crops)

CUSTOM_DIR = ROOT / 'models/patchcore/custom'
TRAIN_DIR = ROOT / 'data/custom_training'
BACKBONE = ROOT / 'models/patchcore/backbone/resnet18-f37072fd.pth'
NAME_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{1,39}$')
MIN_IMAGES, MAX_IMAGES = 20, 400
THRESHOLD_MARGIN = 1.10
EXTENSIONS = {'image/jpeg': '.jpg', 'image/png': '.png'}


def valid_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise ValueError('Model name must be 2-40 characters: lowercase letters, digits, hyphen or underscore.')
    return name


def artifact_path(name: str) -> Path:
    return CUSTOM_DIR / valid_name(name) / 'casting_custom.pt'


def list_models():
    models = []
    for report in sorted(CUSTOM_DIR.glob('*/report.json')) if CUSTOM_DIR.is_dir() else []:
        data = json.loads(report.read_text(encoding='utf-8'))
        models.append({k: data[k] for k in ('name', 'created_at', 'threshold', 'train_images', 'heldout_images', 'bank_patches',
                                           'heldout_score_max', 'artifact_sha256')})
    return models


def stored_images(name: str):
    folder = TRAIN_DIR / valid_name(name)
    return sorted(p for p in folder.glob('*') if p.suffix in EXTENSIONS.values()) if folder.is_dir() else []


def add_image(name: str, content: bytes, media_type: str) -> dict:
    valid_name(name)
    if artifact_path(name).is_file():
        raise FileExistsError('This model is already trained; choose a new name to train another.')
    folder = TRAIN_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(content).hexdigest()
    target = folder / (sha + EXTENSIONS[media_type])
    existing = stored_images(name)
    duplicate = target.is_file()
    if not duplicate and len(existing) >= MAX_IMAGES:
        raise ValueError(f'At most {MAX_IMAGES} training images per model.')
    if not duplicate:
        target.write_bytes(content)
    return {'sha256': sha, 'duplicate': duplicate, 'count': len(stored_images(name))}


def _patch_features(extractor, frame):
    return extract_patch_features(extractor, square_crops(frame)).flatten(0, 1)


def train_custom(name: str, device: str = 'cuda:0', seed: int = 42) -> dict:
    began = time.time()
    valid_name(name)
    target = artifact_path(name)
    if target.is_file():
        raise FileExistsError('This model is already trained; choose a new name to train another.')
    files = stored_images(name)
    if len(files) < MIN_IMAGES:
        raise ValueError(f'Need at least {MIN_IMAGES} distinct normal images; {len(files)} stored.')
    # Content-hash order makes the split deterministic and independent of upload order.
    files = sorted(files, key=lambda p: p.stem)
    held_files = files[::5]
    train_files = [p for p in files if p not in set(held_files)]
    backbone_digest = hashlib.sha256(BACKBONE.read_bytes()).hexdigest()
    if not backbone_digest.startswith('f37072fd'):
        raise ValueError('Verified official pretrained backbone is required.')
    extractor = load_feature_extractor(BACKBONE, device)

    def read(path):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot decode stored image {path.name}')
        return frame

    features = [_patch_features(extractor, read(p)).cpu() for p in train_files]
    bank, coreset = greedy_coreset(torch.cat(features).to(device), seed=seed)
    del features
    held_scores = []
    for path in held_files:
        patches = _patch_features(extractor, read(path))
        distances = nearest_patch_distances(patches, bank)
        held_scores.append(float(distances.max()))
    threshold = float(max(held_scores) * THRESHOLD_MARGIN)
    scores = np.array(held_scores)
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    metadata = {'method': 'compact_patchcore_resnet18', 'data_source': 'custom_normal_images', 'brake_disc_validated': False,
                'custom_name': name, 'seed': seed, 'training_normal_images': len(train_files), 'training_defect_images': 0,
                'heldout_normal_images': len(held_files), 'train_sha256': [p.stem for p in train_files],
                'heldout_sha256': [p.stem for p in held_files], 'coreset': coreset, 'threshold': threshold,
                'threshold_selection': f'Maximum held-out NORMAL score x {THRESHOLD_MARGIN}. No defect images were used.',
                'image_score': 'Maximum nearest normal-patch Euclidean distance over square crops',
                'limitations': 'Few-shot, normal-only model for one casting. False-alarm rate is estimated from a small held-out '
                               'sample and detection of real defects is unmeasured. Scores are not probabilities.'}
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 1, 'bank': bank.cpu().float(), 'threshold': threshold,
                'backbone_path': BACKBONE.relative_to(ROOT).as_posix(), 'backbone_sha256': backbone_digest,
                'metadata': metadata}, target)
    report = {'name': name, 'created_at': now, 'threshold': threshold, 'train_images': len(train_files),
              'heldout_images': len(held_files), 'bank_patches': int(len(bank)),
              'heldout_scores': [round(v, 4) for v in held_scores], 'heldout_score_max': round(float(scores.max()), 4),
              'heldout_score_mean': round(float(scores.mean()), 4), 'threshold_margin': THRESHOLD_MARGIN,
              'artifact_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'seconds': round(time.time() - began, 1),
              'metadata': metadata}
    (target.parent / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
