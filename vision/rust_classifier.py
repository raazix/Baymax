"""Corrosion classifier runtime (ResNet18, two sigmoid heads: corrosion present, severe corrosion).

Trained by scripts/train_rust.py on the cleaned Roboflow "Rust Detection" set (CC BY 4.0). The artifact hash must match
models/rust/evaluation.json, so the runtime only serves the evaluated model. A class-activation map from the corrosion
head localises the evidence; it is a coarse 9x9 attribution, not a segmentation mask.
"""
import hashlib
import json
import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'models/rust/rust_classifier.pt'
EVALUATION = ROOT / 'models/rust/evaluation.json'
DOMAINS = ROOT / 'models/rust/domain_thresholds.json'
MIN_COLOURFULNESS = 2.0   # rust is recognised largely by colour; greyscale images are out of domain


def colourfulness(frame):
    b, g, r = (frame[..., i].astype(np.float32) for i in range(3))
    return float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)))


class RustUnavailable(RuntimeError):
    pass


class RustClassifier:
    def __init__(self, artifact=ARTIFACT):
        self.path = Path(artifact)
        self.lock = Lock()
        self.model = self.meta = self.digest = None
        requested = os.getenv('LINEGUARD_MODEL_DEVICE', 'cpu')
        self.device = f'cuda:{requested}' if requested.isdigit() else requested

    def status(self):
        evaluation = json.loads(EVALUATION.read_text(encoding='utf-8')) if EVALUATION.is_file() else {}
        test = (evaluation.get('test') or {}).get('per_source_photo') or {}
        return {'configured': self.path.is_file(), 'loaded': self.model is not None, 'task': 'corrosion_classification',
                'thresholds': evaluation.get('thresholds'), 'artifact_sha256': evaluation.get('artifact_sha256'),
                'test_corrosion_auroc_per_photo': test.get('corrosion_auroc'), 'brake_disc_validated': False,
                'dataset': 'Roboflow Rust Detection v1 (CC BY 4.0), mosaics removed, grouped re-split'}

    def _load(self):
        if not self.path.is_file() or not EVALUATION.is_file():
            raise RustUnavailable('Corrosion model is not trained; run scripts/prepare_rust.py and scripts/train_rust.py.')
        import io
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18
        payload = self.path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != json.loads(EVALUATION.read_text(encoding='utf-8')).get('artifact_sha256'):
            raise RustUnavailable('Corrosion model differs from its evaluated artifact; re-evaluate before use.')
        artifact = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if artifact.get('format_version') != 1 or artifact.get('heads') != ['corrosion', 'severe']:
            raise RustUnavailable('Unsupported corrosion model artifact.')
        model = resnet18()
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(artifact['state_dict'])
        self.model = model.eval().to(self.device)
        self.meta, self.digest = artifact, digest

    def domain_threshold(self, domain):
        if not domain or not DOMAINS.is_file():
            return None
        table = json.loads(DOMAINS.read_text(encoding='utf-8'))
        if table.get('artifact_sha256') != self.digest:
            return None                                   # thresholds belong to a different model artifact
        return table.get(domain)

    def detect(self, frame, domain=None):
        import torch
        if colourfulness(frame) < MIN_COLOURFULNESS:
            return {'skipped': True, 'reason': 'greyscale image: the corrosion check needs colour', 'corrosion': False, 'severe': False,
                    'grade': None, 'corrosion_probability': None, 'severe_probability': None, 'thresholds': None,
                    'activation_map': None, 'model_sha256': self.digest, 'brake_disc_validated': False, 'note': 'Not assessed.'}
        with self.lock:
            if self.model is None:
                self._load()
            size, meta = self.meta['input_size'], self.meta
            rgb = cv2.cvtColor(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            x = torch.from_numpy(((rgb - np.array(meta['mean'], np.float32)) / np.array(meta['std'], np.float32)).transpose(2, 0, 1)[None]).to(self.device)
            features = {}
            hook = self.model.layer4.register_forward_hook(lambda _m, _i, out: features.__setitem__('map', out))
            try:
                with torch.no_grad():
                    probs = torch.sigmoid(self.model(x).float())[0].cpu().numpy()
            finally:
                hook.remove()
            weights = self.model.fc.weight[0].detach()
            cam = torch.relu(torch.einsum('c,chw->hw', weights, features['map'][0].float())).cpu().numpy()
            cam = cam / cam.max() if cam.max() > 0 else cam
        thresholds = dict(meta['thresholds'])
        calibration = self.domain_threshold(domain)
        if calibration:
            thresholds = {'corrosion': calibration['corrosion'], 'severe': None, 'domain': domain, 'rule': calibration['rule']}
        corrosion = bool(probs[0] >= thresholds['corrosion'])
        # The severe head is only trusted on the domain it was validated on (general rust photos).
        severe = corrosion and thresholds['severe'] is not None and bool(probs[1] >= thresholds['severe'])
        calibrated_corrosion = calibrated_severe = None
        calibration_note = 'No matching validation calibration for this model/domain.'
        if domain is None:
            from analytics.calibration import platt_probability, section
            calibration = section('corrosion', self.digest)
            if calibration:
                calibrated_corrosion = round(platt_probability(float(probs[0]), calibration['corrosion']['params']), 4)
                calibrated_severe = round(platt_probability(float(probs[1]), calibration['severe']['params']), 4)
                calibration_note = 'Platt-calibrated on held-out general rust photos; not valid for brake discs or other domains.'
        return {'skipped': False, 'corrosion_probability': round(float(probs[0]), 4), 'severe_probability': round(float(probs[1]), 4),
                'calibrated_corrosion_probability': calibrated_corrosion, 'calibrated_severe_probability': calibrated_severe,
                'probability_calibration_note': calibration_note,
                'corrosion': corrosion, 'severe': severe, 'grade': 'severe' if severe else 'corrosion' if corrosion else None,
                'thresholds': thresholds, 'activation_map': [[round(float(v), 3) for v in row] for row in cam],
                'model_sha256': self.digest, 'brake_disc_validated': False,
                'note': 'General corrosion photos; probabilities are uncalibrated and only gate the finding, never its severity.'}
