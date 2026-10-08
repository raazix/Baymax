"""Exact TreeSHAP through XGBoost's native contribution implementation."""
import numpy as np
from xgboost import DMatrix


def explain_prediction(booster, features, feature_names, class_index):
    matrix = DMatrix(np.asarray(features, dtype=float).reshape(1, -1), feature_names=list(feature_names))
    contributions = booster.predict(matrix, pred_contribs=True, strict_shape=True)[0, class_index]
    margin = float(booster.predict(matrix, output_margin=True, strict_shape=True)[0, class_index])
    base = float(contributions[-1])
    values = [dict(feature=name, value=float(value), contribution=float(contribution))
              for name, value, contribution in zip(feature_names, features, contributions[:-1])]
    return {'method': 'exact TreeSHAP (XGBoost native pred_contribs)',
            'units': 'raw class margin (pre-softmax); not probability percentages',
            'base_value': base, 'prediction_margin': margin,
            'additivity_residual': float(margin - contributions.sum()),
            'feature_contributions': sorted(values, key=lambda row: abs(row['contribution']), reverse=True)}
