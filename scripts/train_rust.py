"""Train the corrosion classifier on the cleaned Rust Detection split (see scripts/prepare_rust.py).

ResNet18 initialised from the official ImageNet weights already pinned in the project, with two sigmoid heads:
corrosion present, and severe corrosion (trained only on corrosion-positive images). Epoch selection uses validation
only; the untouched test split is scored once at the end. Metrics are reported per image and per source photo
(augmented variants of one photo averaged), because variants of the same photo are correlated.

Outputs: models/rust/rust_classifier.pt (weights + thresholds + provenance), models/rust/evaluation.json.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/processed/rust/manifest.json'
BACKBONE = ROOT / 'models/patchcore/backbone/resnet18-f37072fd.pth'
OUT = ROOT / 'models/rust'
SIZE = 288
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


CACHE_SIZE = 352
_cache = {}


def decoded(path):
    """Decode each image once (RGB, 352 px) and keep it in memory; augmentation then works on the cached array."""
    if path not in _cache:
        image = cv2.cvtColor(cv2.imread(str(ROOT / path)), cv2.COLOR_BGR2RGB)
        _cache[path] = cv2.resize(image, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
    return _cache[path]


def load(path, train, rng):
    image = decoded(path)
    if train:
        h, w = image.shape[:2]
        scale = rng.uniform(.6, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y, x = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
        image = image[y:y + ch, x:x + cw]
        if rng.random() < .5:
            image = image[:, ::-1]
        image = np.clip(image.astype(np.float32) * rng.uniform(.8, 1.2) + rng.uniform(-20, 20), 0, 255)
    image = cv2.resize(np.ascontiguousarray(image).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return ((image.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def build_model():
    model = resnet18()
    model.load_state_dict(torch.load(BACKBONE, map_location='cpu', weights_only=True))
    model.fc = nn.Linear(512, 2)     # [corrosion, severe]
    return model


def batches(records, batch_size, train, rng):
    order = rng.permutation(len(records)) if train else np.arange(len(records))
    for start in range(0, len(order), batch_size):
        chunk = [records[i] for i in order[start:start + batch_size]]
        x = torch.from_numpy(np.stack([load(r['path'], train, rng) for r in chunk]))
        y = torch.tensor([[float(r['corrosion']), float(r['severe'])] for r in chunk])
        yield x, y, chunk


@torch.no_grad()
def predict(model, records, device):
    model.eval()
    probs = []
    rng = np.random.default_rng(0)
    for x, _, _ in batches(records, 64, False, rng):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
            probs.append(torch.sigmoid(model(x.to(device)).float()).cpu().numpy())
    return np.concatenate(probs)


def by_source(records, probs):
    groups = defaultdict(list)
    for r, p in zip(records, probs):
        groups[r['source']].append((r, p))
    out = []
    for items in groups.values():
        out.append(({'corrosion': np.mean([r['corrosion'] for r, _ in items]) >= .5, 'severe': np.mean([r['severe'] for r, _ in items]) >= .5},
                    np.mean([p for _, p in items], axis=0)))
    return [r for r, _ in out], np.array([p for _, p in out])


def metrics(records, probs, thresholds=None):
    y = np.array([r['corrosion'] for r in records], int)
    result = {'n': len(records), 'positives': int(y.sum()), 'negatives': int((1 - y).sum()),
              'corrosion_auroc': float(roc_auc_score(y, probs[:, 0])) if 0 < y.sum() < len(y) else None}
    positives = [i for i, r in enumerate(records) if r['corrosion']]
    ys = np.array([records[i]['severe'] for i in positives], int)
    result['severe_auroc_among_corroded'] = float(roc_auc_score(ys, probs[positives, 1])) if 0 < ys.sum() < len(ys) else None
    if thresholds:
        tc, ts = thresholds
        pred = probs[:, 0] >= tc
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        result |= {'corrosion_threshold': tc, 'corrosion_recall': tp / max(tp + fn, 1), 'corrosion_specificity': tn / max(tn + fp, 1),
                   'corrosion_precision': tp / max(tp + fp, 1), 'corrosion_confusion_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)]}
        sp = probs[positives, 1] >= ts
        stn, sfp, sfn, stp = confusion_matrix(ys, sp, labels=[0, 1]).ravel()
        result |= {'severe_threshold': ts, 'severe_recall': stp / max(stp + sfn, 1), 'severe_specificity': stn / max(stn + sfp, 1)}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}


def youden(y, p):
    """Validation threshold maximising sensitivity + specificity (balanced; negatives are rare)."""
    best = (-1, .5)
    for t in np.unique(np.round(p, 3)):
        pred = p >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        best = max(best, (sens + spec, float(t)))
    return best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if (OUT / 'rust_classifier.pt').exists():
        raise SystemExit('models/rust/rust_classifier.pt exists; move it aside before retraining (reports are preserved).')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    split = {name: [r for r in manifest['records'] if r['split'] == name] for name in ('train', 'validation', 'test')}
    device = args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'
    model = build_model().to(device)
    train = split['train']
    neg = sum(not r['corrosion'] for r in train); pos = len(train) - neg
    corrosion_loss = nn.BCEWithLogitsLoss(reduction='none')
    weights = torch.tensor([1.0, pos / max(neg, 1)])   # [weight for positives, weight for rare negatives]
    optimizer = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if not n.startswith('fc')], 'lr': 1e-4},
                                   {'params': model.fc.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.startswith('cuda'))
    best, history, started = None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for x, y, _ in batches(train, args.batch, True, rng):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
                logits = model(x).float()
            w = torch.where(y[:, 0] > 0, weights[0], weights[1]).to(device)
            loss = (corrosion_loss(logits[:, 0], y[:, 0]) * w).mean()
            mask = y[:, 0] > 0
            if mask.any():
                loss = loss + corrosion_loss(logits[mask, 1], y[mask, 1]).mean()
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item() * len(x)
        scheduler.step()
        val = metrics(split['validation'], predict(model, split['validation'], device))
        score = (val['corrosion_auroc'] or 0) + (val['severe_auroc_among_corroded'] or 0)
        history.append({'epoch': epoch, 'train_loss': round(total / len(train), 4), **val})
        print(f"epoch {epoch:2d} loss {total / len(train):.4f} val corrosion AUROC {val['corrosion_auroc']} severe AUROC {val['severe_auroc_among_corroded']}", flush=True)
        if best is None or score > best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    val_probs = predict(model, split['validation'], device)
    yv = np.array([r['corrosion'] for r in split['validation']], int)
    corr_t = youden(yv, val_probs[:, 0])
    positives = [i for i, r in enumerate(split['validation']) if r['corrosion']]
    sev_t = youden(np.array([split['validation'][i]['severe'] for i in positives], int), val_probs[positives, 1])
    thresholds = (corr_t, sev_t)
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = {'format_version': 1, 'state_dict': best[2], 'heads': ['corrosion', 'severe'], 'input_size': SIZE,
                'mean': MEAN.tolist(), 'std': STD.tolist(), 'thresholds': {'corrosion': corr_t, 'severe': sev_t},
                'backbone_sha256': hashlib.sha256(BACKBONE.read_bytes()).hexdigest(), 'selected_epoch': best[1],
                'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
    torch.save(artifact, OUT / 'rust_classifier.pt')
    digest = hashlib.sha256((OUT / 'rust_classifier.pt').read_bytes()).hexdigest()
    # The artifact and thresholds are frozen before the test split is scored.
    test_probs = predict(model, split['test'], device)
    report = {'dataset': manifest['dataset'], 'license': manifest['license'], 'artifact_sha256': digest,
              'selected_epoch': best[1], 'thresholds': artifact['thresholds'], 'threshold_rule': 'Youden J on validation (balanced sensitivity + specificity)',
              'validation': {'per_image': metrics(split['validation'], val_probs, thresholds),
                             'per_source_photo': metrics(*by_source(split['validation'], val_probs), thresholds)},
              'test': {'per_image': metrics(split['test'], test_probs, thresholds),
                       'per_source_photo': metrics(*by_source(split['test'], test_probs), thresholds)},
              'history': history, 'seconds': round(time.time() - started),
              'limitations': 'Image-level tags from a public Roboflow project (noisy; mosaics removed heuristically); general corrosion photos, '
                             'not automotive brake discs. Severe vs not-severe only: mild/moderate are too rare to learn separately.'}
    (OUT / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('selected_epoch', 'thresholds', 'validation', 'test')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
