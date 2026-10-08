"""Compose trained synthetic analytics and pin artifact identities in evidence."""
import hashlib
import os
from pathlib import Path
from analytics.demo import analyze as demo_analyze

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {'xgboost_rca': ROOT / 'models/xgboost/model.ubj',
             'rca_metadata': ROOT / 'models/xgboost/metadata.json',
             'plsr': ROOT / 'models/plsr/forecast.joblib'}


class AnalyticsUnavailable(RuntimeError):
    pass


def artifact_paths():
    paths = dict(ARTIFACTS)
    if os.getenv('LINEGUARD_RCA_MODEL_DIR'):
        directory = Path(os.environ['LINEGUARD_RCA_MODEL_DIR']).resolve()
        paths.update(xgboost_rca=directory / 'model.ubj', rca_metadata=directory / 'metadata.json')
    return paths


def model_versions():
    paths = artifact_paths()
    if not all(path.is_file() for path in paths.values()):
        return {'analytics': 'demo-heuristic-v1'}
    return {'analytics': 'trained-synthetic-v1',
            **{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}}


def status():
    return {name: {'configured': path.is_file(), 'production_validated': False,
                   'scope': 'synthetic process data',
                   **({'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else {})}
            for name, path in {**artifact_paths(), 'pcr': ROOT / 'models/pcr/forecast.joblib'}.items()
            if name != 'rca_metadata'}


def analyze(telemetry: dict, seed: int, versions: dict | None = None):
    versions = versions or model_versions()
    if versions.get('analytics') == 'demo-heuristic-v1':
        result = demo_analyze(telemetry, seed)
        result['availability_note'] = 'Explicit legacy/demo heuristic; no trained analytics used.'
        result['model_versions'] = {'analytics': 'demo-heuristic-v1'}
        return result
    if versions.get('analytics') != 'trained-synthetic-v1':
        raise AnalyticsUnavailable('Unknown analytics version; refusing to substitute another model.')
    try:
        for name, path in artifact_paths().items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != versions.get(name):
                raise AnalyticsUnavailable('Pinned analytics artifacts changed; original evidence cannot be reproduced.')
        from analytics.xgboost_rca import analyze_rca
        from analytics.plsr_forecast import analyze_forecast
        rca = analyze_rca(telemetry)
        forecast = analyze_forecast(telemetry, seed)
        if rca['provenance']['model_sha256'] != versions['xgboost_rca'] or forecast['forecast']['artifact_sha256'] != versions['plsr']:
            raise AnalyticsUnavailable('Loaded models do not match pinned evidence.')
        for name, path in artifact_paths().items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != versions.get(name):
                raise AnalyticsUnavailable('Analytics artifacts changed during inference; retry with original versions.')
        return {'status': 'synthetic_demonstration', 'rca': rca,
                **forecast, 'model_versions': versions, 'production_validated': False}
    except AnalyticsUnavailable:
        raise
    except (OSError, ImportError, RuntimeError, ValueError, EOFError) as exc:
        raise AnalyticsUnavailable('Trained analytics unavailable; restore pinned artifacts and dependencies.') from exc
