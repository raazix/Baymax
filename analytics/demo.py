import numpy as np

def analyze(telemetry: dict, seed: int) -> dict:
    temperature = max(0, (telemetry['temperature_c'] - 700) / 55)
    vibration = max(0, (telemetry['vibration_mm_s'] - 2) / 4)
    pressure = abs(telemetry['pressure_bar'] - 100) / 25
    terms = {'temperature_c': temperature * .45, 'vibration_mm_s': vibration * .22, 'pressure_bar': pressure * .15}
    risk = float(np.clip(.08 + sum(terms.values()), 0, 1))
    rng = np.random.default_rng(seed)
    samples = np.clip(rng.normal(risk, .055, 10_000), 0, 1)
    ranked = sorted(terms.items(), key=lambda item: item[1], reverse=True)
    return {'status': 'synthetic_demonstration', 'rca': {
        'hypothesis': 'thermal_process_drift' if temperature > .3 else 'surface_or_tooling_issue',
        'confidence': None, 'confidence_note': 'No calibrated RCA model trained.',
        'feature_contributions': [{'feature': k, 'heuristic_risk_contribution': round(v, 4)} for k, v in ranked],
        'method': 'demo heuristic; not XGBoost or TreeSHAP', 'causality_established': False},
        'forecast': {'risk': round(risk, 4), 'horizon': 'next lot',
            'method': 'synthetic heuristic; PLSR pending', 'pcr_role': 'benchmark only'},
        'uncertainty': {'mean': round(float(samples.mean()), 4),
            'interval_95': [round(float(x), 4) for x in np.quantile(samples, [.025, .975])],
            'interval_type': 'simulated predictive interval; not a validated confidence interval',
            'tolerance': .5, 'breach_probability': round(float(np.mean(samples > .5)), 4),
            'simulations': 10_000, 'seed': seed, 'assumption': 'Clipped normal risk distribution, standard deviation 0.055'}}
