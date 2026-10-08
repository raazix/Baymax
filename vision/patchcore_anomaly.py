"""Compact PatchCore casting proxy with local-only frozen ImageNet features."""
import hashlib
import json
import math
import os
from functools import wraps
from pathlib import Path
from threading import Lock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIM = 384
GRID_SIZE = 28
MAX_CROPS = 16


class PatchCoreUnavailable(RuntimeError):
    pass


def plan_square_crops(frame):
    """None for near-square frames; otherwise overlapping square crop offsets covering the long axis."""
    height, width = frame.shape[:2]
    if .9 <= width / height <= 1.1:
        return None
    side, long_edge = min(height, width), max(height, width)
    needed = max(2, math.ceil((long_edge - side) / (side * .5)) + 1)
    if needed > MAX_CROPS:
        raise ValueError(f'Aspect ratio too extreme for sliced inference (needs {needed} crops, limit {MAX_CROPS})')
    offsets = np.linspace(0, long_edge - side, needed).round().astype(int)
    return {'offsets': offsets, 'side': int(side), 'horizontal': width > height, 'needed': int(needed),
            'cell': side / GRID_SIZE, 'total': int(round(long_edge / (side / GRID_SIZE)))}


def square_crops(frame):
    plan = plan_square_crops(frame)
    if plan is None:
        return [frame]
    side = plan['side']
    return [frame[:, o:o + side] if plan['horizontal'] else frame[o:o + side, :] for o in plan['offsets']]


def _inference(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch
        with torch.inference_mode():
            return function(*args, **kwargs)
    return wrapped


def FrozenFeatures(backbone):
    """Construct a frozen feature module lazily; base API does not import Torch."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class FeatureNetwork(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
            self.requires_grad_(False)
            self.eval()

        def forward(self, batch):
            layer2 = self.layer2(self.layer1(self.stem(batch)))
            layer3 = self.layer3(layer2)
            layer2 = F.avg_pool2d(layer2, 3, stride=1, padding=1)
            layer3 = F.avg_pool2d(layer3, 3, stride=1, padding=1)
            layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode='bilinear', align_corners=False)
            return torch.cat([layer2, layer3], dim=1).flatten(2).transpose(1, 2).contiguous()
    return FeatureNetwork(backbone)


def load_feature_extractor(backbone_path, device='cpu'):
    """Never downloads: an explicit local torchvision ResNet18 state dict is required."""
    path = Path(backbone_path)
    if not path.is_file():
        raise PatchCoreUnavailable(f'Local ImageNet backbone missing: {path}')
    try:
        import torch
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        return FrozenFeatures(backbone).to(device).eval()
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        raise PatchCoreUnavailable('Cannot load the local ResNet18 ImageNet backbone') from exc


def preprocess_frames(frames_bgr, device='cpu'):
    """BGR uint8 to ImageNet normalized RGB at 224x224; no crop."""
    import torch
    from torch.nn import functional as F
    rows = []
    for frame in frames_bgr:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
            raise ValueError('Expected nonempty BGR uint8 images with three channels')
        image = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.
        rows.append(F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False, antialias=True))
    if not rows:
        raise ValueError('At least one image is required')
    batch = torch.cat(rows).to(device)
    mean = batch.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = batch.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return (batch - mean) / std


@_inference
def extract_patch_features(extractor, frames_bgr):
    import torch
    device = next(extractor.parameters()).device
    features = extractor(preprocess_frames(frames_bgr, device))
    if features.shape[1:] != (GRID_SIZE * GRID_SIZE, FEATURE_DIM) or not torch.isfinite(features).all():
        raise ValueError('Expected finite ResNet18 layer2/layer3 patch embeddings [B,784,384]')
    return features


@_inference
def nearest_patch_distances(features, bank, chunk_size=256, bank_chunk_size=4096):
    """Exact Euclidean nearest neighbours using bounded matrix multiplication."""
    import torch
    if features.ndim != 2 or bank.ndim != 2 or features.shape[1] != bank.shape[1] or not len(bank) or chunk_size < 1 or bank_chunk_size < 1:
        raise ValueError('Expected compatible nonempty two-dimensional bank and positive chunks')
    if not torch.isfinite(features).all() or not torch.isfinite(bank).all():
        raise ValueError('Patch embeddings must be finite')
    if bank.device != features.device:
        raise ValueError('Features and bank must share device')
    features, bank = features.float(), bank.float()
    output = []
    bank_norm = (bank * bank).sum(1)
    for begin in range(0, len(features), chunk_size):
        query = features[begin:begin + chunk_size]
        query_norm = (query * query).sum(1, keepdim=True)
        best = torch.full((len(query),), float('inf'), device=query.device)
        for offset in range(0, len(bank), bank_chunk_size):
            reference = bank[offset:offset + bank_chunk_size]
            squared = query_norm + bank_norm[offset:offset + bank_chunk_size].unsqueeze(0) - 2 * query @ reference.T
            best = torch.minimum(best, squared.clamp_min_(0).amin(1))
        output.append(best.sqrt())
    return torch.cat(output) if output else features.new_empty((0,))


@_inference
def greedy_coreset(features, max_points=1024, projection_dim=64, seed=42, candidate_limit=50000):
    """Approximate farthest-point coreset in a seeded random projected space."""
    import torch
    if features.ndim != 2 or not len(features) or max_points < 1 or projection_dim < 1 or candidate_limit < 1 or candidate_limit > 50000:
        raise ValueError('Invalid features or coreset configuration; candidate limit must be <=50000')
    if not torch.isfinite(features).all():
        raise ValueError('Features must be finite')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    count = len(features)
    candidate_indices = torch.randperm(count, generator=generator)[:candidate_limit] if count > candidate_limit else torch.arange(count)
    candidates = features[candidate_indices.to(features.device)].float()
    projection = torch.randn(features.shape[1], projection_dim, generator=generator).to(features.device) / math.sqrt(projection_dim)
    projected = candidates @ projection
    first = int(((projected - projected.mean(0)) ** 2).sum(1).argmax())
    selected = []
    best = torch.full((len(candidates),), float('inf'), device=features.device)
    current = first
    for _ in range(min(max_points, len(candidates))):
        selected.append(current)
        distance = ((projected - projected[current]) ** 2).sum(1)
        best = torch.minimum(best, distance)
        best[selected] = -1
        current = int(best.argmax())
    bank = candidates[selected].contiguous()
    return bank, {'method': 'approximate greedy farthest-point random projection', 'seed': seed,
                  'input_patches': count, 'candidate_patches': len(candidates), 'candidate_limit': candidate_limit,
                  'candidate_pool_subsampled': count > candidate_limit, 'projection_dim': projection_dim,
                  'selected_patches': len(bank), 'projection_max_squared_covering_distance': float(best.clamp_min(0).max()),
                  'approximation': 'Seeded random candidate subsampling when needed, then Gaussian projected-space greedy selection; original embeddings retained.'}


class PatchCoreDetector:
    def __init__(self, artifact_path=None):
        selected_path = artifact_path or os.getenv('LINEGUARD_PATCHCORE_WEIGHTS')
        self.expected_digest = None
        self.evaluation_reference = None
        registry = ROOT / 'models/patchcore/active.json'
        registered_device = 'cpu'
        if selected_path is None and registry.is_file():
            active = json.loads(registry.read_text(encoding='utf-8'))
            selected_path = (ROOT / active['weights']).resolve()
            if not selected_path.is_relative_to(ROOT):
                raise ValueError('Registered PatchCore artifact must remain in the workspace')
            self.expected_digest = active['weights_sha256']
            self.evaluation_reference = active.get('evaluation')
            registered_device = active.get('device', 'cpu')
        self.path = Path(selected_path or ROOT / 'models/patchcore/casting_proxy.pt')
        requested_device = os.getenv('LINEGUARD_MODEL_DEVICE', registered_device)
        self.device = f'cuda:{requested_device}' if requested_device.isdigit() else requested_device
        self.lock = Lock()
        self.extractor = None
        self.bank = None
        self.metadata = None
        self.threshold = None
        self.digest = None

    def status(self):
        return {'configured': self.path.is_file(), 'loaded': self.extractor is not None,
                'task': 'casting_proxy_anomaly_detection', 'brake_disc_validated': False,
                'model_sha256': self.digest or self.expected_digest, 'artifact_sha256': self.digest or self.expected_digest,
                'evaluation': self.evaluation_reference, 'threshold': self.threshold, 'device': self.device,
                'memory_bank_patches': None if self.bank is None else int(len(self.bank)),
                'backbone': 'ImageNet ResNet18 frozen layer2/layer3', 'scope': 'casting proxy only',
                'method': 'compact_patchcore_resnet18', 'metrology': None}

    def _load(self):
        if not self.path.is_file():
            raise PatchCoreUnavailable('Train the casting PatchCore proxy and configure its local artifact')
        try:
            import torch
            payload = self.path.read_bytes()
            if self.expected_digest and hashlib.sha256(payload).hexdigest() != self.expected_digest:
                raise ValueError('Registered PatchCore artifact changed since calibration/evaluation')
            import io
            artifact = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
            if artifact.get('format_version') != 1:
                raise ValueError('Unsupported PatchCore artifact format; expected format_version=1')
            bank = artifact['bank']
            threshold = float(artifact['threshold'])
            metadata = artifact['metadata']
            if not isinstance(bank, torch.Tensor) or bank.ndim != 2 or bank.shape[1] != FEATURE_DIM or not 0 < bank.shape[0] <= 50000 or not torch.isfinite(bank).all():
                raise ValueError('Invalid finite normal-patch memory bank')
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError('Threshold must be a positive finite distance')
            if metadata.get('method') != 'compact_patchcore_resnet18' or metadata.get('data_source') not in ('casting_proxy', 'custom_normal_images', 'mpdd_proxy') or metadata.get('brake_disc_validated') is not False:
                raise ValueError('Invalid casting proxy provenance')
            backbone_path = (ROOT / artifact['backbone_path']).resolve()
            if not backbone_path.is_relative_to(ROOT):
                raise ValueError('Backbone must be local to the workspace')
            if hashlib.sha256(backbone_path.read_bytes()).hexdigest() != artifact['backbone_sha256']:
                raise ValueError('Backbone hash does not match trained artifact')
            extractor = load_feature_extractor(backbone_path, self.device)
            bank = bank.to(device=self.device, dtype=torch.float32)
            self.extractor, self.bank = extractor, bank
            self.threshold, self.metadata = threshold, metadata
            self.digest = hashlib.sha256(payload).hexdigest()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
            raise PatchCoreUnavailable(f'Casting PatchCore unavailable: {exc}') from exc

    def _score_square(self, frame):
        patches = extract_patch_features(self.extractor, [frame])[0]
        return nearest_patch_distances(patches, self.bank).reshape(GRID_SIZE, GRID_SIZE).cpu().numpy()

    def _grid(self, frame):
        """Anomaly grid for any aspect ratio. Near-square frames take the original single-pass path; others are sliced
        into overlapping square crops (each scored exactly as a training-style square) and stitched at true aspect."""
        plan = plan_square_crops(frame)
        if plan is None:
            return self._score_square(frame), None
        total, cell = plan['total'], plan['cell']
        merged = np.zeros((GRID_SIZE, total), dtype=np.float32)
        for offset, crop in zip(plan['offsets'], square_crops(frame)):
            grid = self._score_square(crop)
            if not plan['horizontal']: grid = grid.T
            start = int(round(offset / cell)); stop = min(total, start + GRID_SIZE)
            merged[:, start:stop] = np.maximum(merged[:, start:stop], grid[:, :stop - start])
        merged = merged if plan['horizontal'] else merged.T
        return merged, {'crops': plan['needed'], 'crop_side_px': plan['side'],
                        'note': 'Non-square image sliced into overlapping square crops; score is the maximum over crops.'}

    def detect(self, frame):
        with self.lock:
            if self.extractor is None:
                self._load()
            grid, tiling = self._grid(frame)
            score = float(grid.max())
            return {'source': 'trained_mpdd_patchcore_proxy' if self.metadata.get('data_source') == 'mpdd_proxy' else 'trained_casting_patchcore_proxy', 'method': 'compact_patchcore_resnet18',
                    'label': 'ANOMALY_UNCLASSIFIED' if score > self.threshold else 'normal',
                    'anomaly_score': score, 'threshold': self.threshold, 'score_units': 'nearest normal-patch Euclidean feature distance; not a probability',
                    'anomaly_grid': grid.tolist(),
                    'grid_shape': list(grid.shape), 'grid_note': 'Raw patch-distance grid; not a pixel segmentation mask.',
                    'tiling': tiling,
                    'model_sha256': self.digest, 'artifact_sha256': self.digest, 'provenance': self.metadata,
                    'brake_disc_validated': False, 'metrology': None}
