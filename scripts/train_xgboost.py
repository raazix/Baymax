"""Train a reproducible SYNTHETIC process hypothesis classifier."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics.xgboost_rca import FEATURES, MODEL_DIR

CLASSES = ['nominal_process', 'thermal_process_drift', 'pressure_instability', 'tooling_vibration', 'speed_drift']


def generate_history(seed=42, lots=500, parts_per_lot=12):
    rng = np.random.default_rng(seed)
    rows, labels, lot_ids = [], [], []
    offsets = np.asarray([[0, 0, 0, 0], [48, 3, .3, 0], [4, -22, .2, 0],
                          [0, 2, 3.4, 30], [8, 0, .4, 280]])
    for lot in range(lots):
        label = int(rng.integers(len(CLASSES)))
        mean = np.asarray([705, 100, 2, 1200]) + offsets[label]
        mean += rng.normal(0, [10, 5, .5, 55])
        # Slow future drift makes chronological evaluation meaningful.
        mean += (lot / lots) * np.asarray([5, -2, .2, 25])
        for _ in range(parts_per_lot):
            rows.append(mean + rng.normal(0, [5, 3, .3, 30]))
            labels.append(label)
            lot_ids.append(lot)
    return np.asarray(rows), np.asarray(labels), np.asarray(lot_ids)


def train(output=MODEL_DIR, seed=42):
    x, y, lots = generate_history(seed)
    train_mask, test_mask = lots < 400, lots >= 400
    model = XGBClassifier(n_estimators=180, max_depth=3, learning_rate=.05,
                          subsample=.85, colsample_bytree=.9, n_jobs=2,
                          random_state=seed, eval_metric='mlogloss')
    # Names persisted in artifact for exact TreeSHAP and input ordering.
    import pandas as pd
    model.fit(pd.DataFrame(x[train_mask], columns=FEATURES), y[train_mask])
    probabilities = model.predict_proba(pd.DataFrame(x[test_mask], columns=FEATURES))
    predicted = probabilities.argmax(axis=1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / 'model.ubj')
    metadata = {'model_id': 'synthetic-process-rca-v1', 'dataset_status': 'synthetic',
                'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed,
                'classes': CLASSES, 'features': list(FEATURES),
                'model_sha256': hashlib.sha256((output / 'model.ubj').read_bytes()).hexdigest(),
                'dataset_sha256': hashlib.sha256(x.tobytes() + y.tobytes() + lots.tobytes()).hexdigest(),
                'split': {'method': 'chronological, whole lots', 'train_lots': [0, 399],
                          'test_lots': [400, 499], 'train_rows': int(train_mask.sum()),
                          'test_rows': int(test_mask.sum())},
                'training_feature_ranges': {f: [float(x[train_mask, i].min()), float(x[train_mask, i].max())]
                                            for i, f in enumerate(FEATURES)},
                'metrics': {'accuracy': accuracy_score(y[test_mask], predicted),
                            'log_loss': log_loss(y[test_mask], probabilities),
                            'classification_report': classification_report(y[test_mask], predicted, target_names=CLASSES, output_dict=True),
                            'confusion_matrix': confusion_matrix(y[test_mask], predicted).tolist()},
                'limitations': ['Labels are sampled latent synthetic process scenarios, not diagnosed production failures.',
                                'No brake-disc or real-production validation.', 'Probabilities are uncalibrated.',
                                'Spatial fingerprint features are not included; future training requires paired production evidence.']}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(output / 'synthetic_history.npz', features=x, labels=y, lot_ids=lots)
    print(json.dumps(metadata['metrics'], indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=MODEL_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.output, args.seed)
