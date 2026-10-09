"""Optional NEU detection adapter, intentionally separate from rotor metrology."""
import hashlib
import os
import json
from pathlib import Path
from threading import Lock
from core.config import ROOT

class ModelUnavailable(Exception): pass

class ProxyDetector:
    def __init__(self):
        self.model = None
        self.lock = Lock()
        self.digest = None
        self.path = os.getenv('LINEGUARD_PROXY_WEIGHTS')
        self.expected_digest = None
        self.evaluation_reference = None
        self.device = os.getenv('LINEGUARD_MODEL_DEVICE', 'cpu')
        registry = ROOT / 'models/yolo/active.json'
        if not self.path and registry.is_file():
            active = json.loads(registry.read_text(encoding='utf-8'))
            candidate = (ROOT / active['weights']).resolve()
            if not candidate.is_relative_to(ROOT): raise ValueError('Registered weights must remain in the workspace')
            self.path = str(candidate)
            self.expected_digest = active['weights_sha256']
            self.evaluation_reference = active.get('evaluation')
            self.device = os.getenv('LINEGUARD_MODEL_DEVICE', active.get('device', 'cpu'))

    def status(self):
        return {'configured': bool(self.path and Path(self.path).is_file()), 'loaded': self.model is not None,
                'task': 'steel_surface_detection_proxy', 'brake_disc_validated': False,
                'model_sha256': self.digest or self.expected_digest, 'evaluation': self.evaluation_reference,
                'device': self.device}

    def detect(self, frame, inference_mode='full', tile_size=512, overlap=.2):
        if inference_mode not in ('full', 'sliced'):
            raise ValueError('inference_mode must be full or sliced')
        with self.lock:
            if not self.path or not Path(self.path).is_file():
                raise ModelUnavailable('Train NEU YOLO detection and set LINEGUARD_PROXY_WEIGHTS to its best.pt file.')
            if self.model is None:
                (ROOT / 'data/yolo-config/Ultralytics').mkdir(parents=True, exist_ok=True)
                os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'data/yolo-config'))
                self.digest = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
                if self.expected_digest and self.digest != self.expected_digest:
                    raise ModelUnavailable('Registered weights changed since evaluation; re-evaluate before activation.')
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.path)
                except (ImportError, OSError, RuntimeError) as error:
                    raise ModelUnavailable('Cannot load proxy model; check ML dependencies and trained weights.') from error
                from scripts.prepare_neu import CLASSES
                names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
                if names != CLASSES or self.model.task != 'detect':
                    self.model = None
                    raise ModelUnavailable('Expected a NEU detector with the six original labels; generic COCO weights are unsuitable.')
            if inference_mode == 'sliced':
                from vision.sahi_inference import sliced_detect
                detections, inference = sliced_detect(self.model, frame, self.device, tile_size, overlap)
            else:
                result = self.model.predict(frame, imgsz=640, conf=.25, device=self.device, verbose=False)[0]
                detections = [{'label': self.model.names[int(box.cls.item())], 'confidence': float(box.conf.item()),
                               'bbox_xyxy_px': box.xyxy[0].tolist()} for box in result.boxes]
                inference = {'mode': 'full', 'engine': 'ultralytics', 'model_input_size': 640, 'confidence_threshold': .25}
            from analytics.calibration import isotonic
            for detection in detections:
                detection['calibrated_confidence'] = isotonic('yolo', detection['confidence'], self.digest)
            return {'source': 'trained_neu_proxy', 'detections': detections,
                    'inference': inference,
                    'model_sha256': self.digest, 'brake_disc_validated': False,
                    'metrology': None, 'note': 'Steel proxy boxes; no masks, physical measurements or rotor severity assigned.'}
