"""Synthetic process hypothesis classifier; does not establish causality."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

FEATURES = ('temperature_c', 'pressure_bar', 'vibration_mm_s', 'machine_speed_rpm')
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'xgboost'


class RCAModelUnavailable(RuntimeError):
    pass


def _load(directory):
    root = Path(directory)
    try:
        model_digest = hashlib.sha256((root / 'model.ubj').read_bytes()).hexdigest()
        metadata_digest = hashlib.sha256((root / 'metadata.json').read_bytes()).hexdigest()
    except OSError as exc:
        raise RCAModelUnavailable(f'RCA model unavailable at {root}: {exc}') from exc
    return _load_verified(directory, model_digest, metadata_digest)


@lru_cache(maxsize=4)
def _load_verified(directory, model_digest, metadata_digest):
    from xgboost import Booster
    root = Path(directory)
    try:
        metadata = json.loads((root / 'metadata.json').read_text())
        model_path = root / 'model.ubj'
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != metadata['model_sha256']:
            raise ValueError('Model hash does not match provenance')
        booster = Booster()
        booster.load_model(model_path)
        return booster, metadata
    except (OSError, ValueError, KeyError) as exc:
        raise RCAModelUnavailable(f'RCA model unavailable at {root}: {exc}') from exc


def analyze_rca(telemetry: dict) -> dict:
    try:
        from xgboost import DMatrix
        from analytics.treeshap_explainer import explain_prediction
    except ImportError as exc:
        raise RCAModelUnavailable('Install XGBoost to use the RCA model') from exc
    try:
        values = np.asarray([float(telemetry[key]) for key in FEATURES])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'RCA requires finite telemetry: {FEATURES}') from exc
    if not np.isfinite(values).all():
        raise ValueError('RCA telemetry must be finite')
    booster, metadata = _load(os.environ.get('LINEGUARD_RCA_MODEL_DIR', str(MODEL_DIR)))
    probabilities = booster.predict(DMatrix(values.reshape(1, -1), feature_names=list(FEATURES)))[0]
    index = int(np.argmax(probabilities))
    explanation = explain_prediction(booster, values, FEATURES, index)
    bounds = metadata['training_feature_ranges']
    out_of_range = [name for name, value in zip(FEATURES, values)
                    if not bounds[name][0] <= value <= bounds[name][1]]
    return {'status': 'synthetic_demonstration', 'hypothesis': metadata['classes'][index],
            'confidence': float(probabilities[index]), 'confidence_calibrated': False,
            'confidence_note': 'Uncalibrated synthetic classifier probability; not causal confidence.',
            'causality_established': False,
            'class_probabilities': dict(zip(metadata['classes'], map(float, probabilities))),
            **explanation, 'method': 'XGBoost + exact TreeSHAP', 'out_of_training_range': out_of_range,
            'provenance': metadata}
