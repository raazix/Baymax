"""PCR benchmark; not a second mandatory production prediction."""
from pathlib import Path
import numpy as np
from analytics.plsr_forecast import ROOT, feature_row, load_artifact

MODEL_PATH = ROOT / 'models/pcr/forecast.joblib'


def predict_benchmark(telemetry: dict, path: Path = MODEL_PATH) -> dict:
    artifact = load_artifact(path)
    row, note = feature_row(telemetry)
    prediction = float(np.clip(artifact['model'].predict(row.reshape(1, -1)).ravel()[0], 0, 1))
    return {'predicted_defect_fraction': prediction, 'method': 'PCR synthetic benchmark',
            'role': 'benchmark only', 'held_out_mae': artifact['metrics']['test_mae'],
            'history_assumption': note, 'production_validated': False}
