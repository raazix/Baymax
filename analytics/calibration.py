"""Artifact-scoped confidence and interval calibration parameters."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'models/calibration/calibration.json'


@lru_cache(maxsize=1)
def _table():
    try:
        return json.loads(PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def section(name: str, artifact_sha256: str | None = None) -> dict | None:
    item = _table().get(name)
    if not isinstance(item, dict):
        return None
    if artifact_sha256 is not None and item.get('artifact_sha256') != artifact_sha256:
        return None
    return item


def isotonic(name: str, score: float, artifact_sha256: str | None):
    item = section(name, artifact_sha256)
    params = (item or {}).get('params') or {}
    x, y = params.get('x'), params.get('y')
    if not x or len(x) != len(y):
        return None
    import numpy as np
    return float(np.interp(score, x, y))


def platt_probability(raw: float, params: dict) -> float:
    import math
    p = min(1 - 1e-6, max(1e-6, float(raw)))
    z = math.log(p / (1 - p))
    z = float(params['a']) * z + float(params['b'])
    return 1 / (1 + math.exp(-max(-40, min(40, z))))
