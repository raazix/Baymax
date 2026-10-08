"""Empirical residual bootstrap, conditional on fixed supplied process telemetry."""
import numpy as np


def simulate(prediction: float, residuals, seed: int, simulations: int = 10_000, tolerance: float = .5) -> dict:
    errors = np.asarray(residuals, dtype=float).ravel()
    if not np.isfinite(prediction) or not 0 <= prediction <= 1:
        raise ValueError('prediction must be a finite fraction in [0, 1]')
    if errors.size < 20 or not np.isfinite(errors).all():
        raise ValueError('At least 20 finite calibration residuals are required')
    if simulations < 100 or not 0 <= tolerance <= 1:
        raise ValueError('Invalid simulation count or tolerance')
    samples = np.clip(prediction + np.random.default_rng(seed).choice(errors, simulations, replace=True), 0, 1)
    return {'mean': float(samples.mean()), 'interval_95': np.quantile(samples, [.025, .975]).tolist(),
            'interval_type': 'simulated predictive interval; not a confidence interval',
            'tolerance': tolerance, 'breach_probability': float(np.mean(samples > tolerance)),
            'simulations': simulations, 'seed': seed, 'calibration_residual_count': int(errors.size),
            'data_source': 'synthetic_process_history', 'production_validated': False,
            'assumption': 'Fixed process inputs; independently bootstrap temporally held-out calibration residuals, then clip to [0,1]. Assumes future errors exchangeable with synthetic calibration errors; excludes model-parameter and input uncertainty.'}


def histogram(prediction: float, residuals, seed: int, simulations: int = 10_000, bins: int = 20) -> dict:
    """Binned view of the exact same seeded samples that simulate() summarises."""
    errors = np.asarray(residuals, dtype=float).ravel()
    samples = np.clip(prediction + np.random.default_rng(seed).choice(errors, simulations, replace=True), 0, 1)
    counts, edges = np.histogram(samples, bins=bins, range=(0, 1))
    return {'edges': edges.round(4).tolist(), 'counts': counts.tolist(), 'simulations': simulations}
