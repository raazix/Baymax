"""Synthetic next-lot defect-fraction regression; never a calibrated probability."""
from functools import lru_cache
from pathlib import Path
import hashlib
from io import BytesIO
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models/plsr/forecast.joblib'
FEATURES = ('temperature_c', 'pressure_bar', 'vibration_mm_s', 'machine_speed_rpm')


class ForecastModelUnavailable(RuntimeError):
    pass


def feature_row(telemetry: dict) -> tuple[np.ndarray, str]:
    def values(item):
        try:
            row = np.array([float(item[key]) for key in FEATURES])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Forecast requires finite telemetry: {FEATURES}') from exc
        if not np.isfinite(row).all():
            raise ValueError('Forecast telemetry must be finite')
        return row
    current = values(telemetry)
    history = telemetry.get('history', [])
    if not isinstance(history, list):
        raise ValueError('history must be a chronological list of prior-lot telemetry')
    prior = [values(item) for item in history[-2:]]
    note = 'observed prior-lot history' if len(prior) == 2 else 'missing prior lots filled with current telemetry; persistence assumption'
    prior = [current] * (2 - len(prior)) + prior
    return np.concatenate([current, prior[-1], np.mean(prior, axis=0)]), note


@lru_cache(maxsize=4)
def _load(digest: str, content: bytes):
    artifact = joblib.load(BytesIO(content))
    artifact['_loaded_sha256'] = digest
    return artifact


def load_artifact(path: Path = MODEL_PATH):
    if not path.is_file():
        raise ForecastModelUnavailable(f'Forecast model unavailable: {path}; run scripts/train_forecast.py')
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ForecastModelUnavailable(f'Forecast model unavailable: {path}') from exc
    return _load(hashlib.sha256(content).hexdigest(), content)


def analyze_forecast(telemetry: dict, seed: int) -> dict:
    from analytics.monte_carlo import simulate
    artifact = load_artifact()
    row, history_note = feature_row(telemetry)
    prediction = float(np.clip(artifact['model'].predict(row.reshape(1, -1)).ravel()[0], 0, 1))
    from analytics.calibration import section
    calibration = section('forecast', artifact['_loaded_sha256'])
    interval_scale = (calibration or {}).get('interval_scale', 1.0)
    uncertainty = simulate(prediction, artifact['residuals'], seed=seed, interval_scale=interval_scale)
    uncertainty['interval_calibration'] = 'synthetic later-lot coverage calibration' if calibration else 'not calibrated for this artifact'
    return {'forecast': {'risk': prediction, 'predicted_defect_fraction': prediction,
        'horizon': 'next lot', 'method': 'trained PLSR on synthetic ordered lots',
        'pcr_role': 'benchmark only', 'calibrated_event_probability': False,
        'data_source': 'synthetic_process_history', 'production_validated': False,
        'history_assumption': history_note, 'model_version': artifact['version'],
        'artifact_sha256': artifact['_loaded_sha256'],
        'held_out_mae': artifact['metrics']['test_mae'],
        'training_feature_bounds': artifact['feature_bounds'],
        'outside_training_range': bool(np.any(row < np.array(artifact['feature_min'])) or np.any(row > np.array(artifact['feature_max'])))},
        'uncertainty': uncertainty}
