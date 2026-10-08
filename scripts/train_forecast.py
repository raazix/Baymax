"""Reproducible synthetic training with strictly chronological holdouts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analytics.plsr_forecast import ROOT, FEATURES, feature_row


def synthetic_lots(seed=2026, count=1800):
    rng = np.random.default_rng(seed)
    state = np.zeros(4)
    telemetry, targets = [], []
    for _ in range(count + 1):
        state = .82 * state + rng.normal(0, .55, 4)
        row = dict(zip(FEATURES, np.array([715, 100, 3, 1200]) + state * [28, 12, 1.4, 100]))
        previous = telemetry[-1] if telemetry else row
        expected = .27 + .003 * (row['temperature_c'] - 715) - .004 * (row['pressure_bar'] - 100) + .06 * (row['vibration_mm_s'] - 3) + .00028 * (previous['machine_speed_rpm'] - 1200)
        targets.append(rng.binomial(250, np.clip(expected, .01, .95)) / 250)
        telemetry.append(row)
    x = np.array([feature_row({**telemetry[i], 'history': telemetry[i-2:i]})[0] for i in range(2, count)])
    y = np.array(targets[3:count+1])
    return x, y


def train(seed=2026):
    x, y = synthetic_lots(seed)
    train_end, calibration_end = int(len(y)*.6), int(len(y)*.8)
    report = {'seed': seed, 'source': 'synthetic ordered lots; no production observations',
              'target': 'observed synthetic next-lot defective count / 250 parts',
              'split': {'train': train_end, 'calibration': calibration_end-train_end, 'test': len(y)-calibration_end},
              'leakage_control': 'Chronological split, no shuffle; scaler/components fitted only on training rows. Calibration residuals exclude test rows.'}
    for name, model in [('plsr', make_pipeline(StandardScaler(), PLSRegression(n_components=4, scale=False))),
                        ('pcr', make_pipeline(StandardScaler(), PCA(n_components=6), LinearRegression()))]:
        model.fit(x[:train_end], y[:train_end])
        cal_prediction = np.clip(model.predict(x[train_end:calibration_end]).ravel(), 0, 1)
        test_prediction = np.clip(model.predict(x[calibration_end:]).ravel(), 0, 1)
        metrics = {'test_mae': float(np.mean(abs(test_prediction-y[calibration_end:]))),
                   'calibration_mae': float(np.mean(abs(cal_prediction-y[train_end:calibration_end]))),
                   'persistence_baseline_test_mae': float(np.mean(abs(y[calibration_end-1:-1]-y[calibration_end:])))}
        artifact = {'model': model, 'version': f'{name}-synthetic-v1-seed-{seed}', 'metrics': metrics,
                    'residuals': y[train_end:calibration_end]-cal_prediction,
                    'feature_min': x[:train_end].min(axis=0).tolist(), 'feature_max': x[:train_end].max(axis=0).tolist(),
                    'feature_bounds': {key: [float(x[:train_end,j].min()), float(x[:train_end,j].max())] for j,key in enumerate(FEATURES)},
                    'provenance': report.copy()}
        folder = ROOT / 'models' / name
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, folder / 'forecast.joblib')
        report[name] = metrics
    (ROOT / 'models/plsr/training_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    train(parser.parse_args().seed)
