"""Deterministic demo SOP recommendations; execution requires an engineer."""
import math

SOP_VERSION = 'demo-sop-v2'
HYPOTHESIS_ACTIONS = {
    'thermal_process_drift': 'Inspect the thermocouple and temperature-control loop; verify calibration against an approved reference.',
    'pressure_instability': 'Inspect mould-pressure sensors, regulators and pneumatic/hydraulic connections; verify sensor calibration and pressure stability.',
    'tooling_vibration': 'Inspect tooling, fixtures, bearings and alignment; verify vibration-sensor calibration and review the vibration trend.',
    'speed_drift': 'Inspect spindle-speed feedback and the encoder; verify calibration and compare recorded speed with the approved process specification.',
}


def _fraction(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and 0 <= result <= 1 else None


def recommend(defects: list[dict], analytics: dict | None) -> dict:
    analytics = analytics if isinstance(analytics, dict) else {}
    rca = analytics.get('rca') or {}
    hypothesis = rca.get('hypothesis') if isinstance(rca, dict) else None
    hypothesis = hypothesis if isinstance(hypothesis, str) else None
    critical = any((defect.get('severity', {}).get('level') if isinstance(defect.get('severity'), dict)
                    else defect.get('severity')) in ('critical', 'CRITICAL') for defect in defects)
    steps = []
    if critical:
        steps.append('Quarantine the affected lot and hold affected parts pending engineer review.')
    elif defects:
        steps.append('Hold the affected part pending engineer review.')
    else:
        steps.append('Continue sampling and monitor process drift.')
    if defects:
        steps.append(HYPOTHESIS_ACTIONS.get(hypothesis,
            'Review the observed defect evidence and process history manually to investigate the cause; no supported process cause has been established.'))
        steps.append('After an approved corrective action, re-inspect a defined sample and record verification evidence.')

    forecast = analytics.get('forecast') or {}
    uncertainty = analytics.get('uncertainty') or {}
    forecast = forecast if isinstance(forecast, dict) else {}
    uncertainty = uncertainty if isinstance(uncertainty, dict) else {}
    fraction = _fraction(forecast.get('predicted_defect_fraction', forecast.get('risk')))
    supporting = {'status': 'synthetic_supporting_evidence', 'production_validated': False,
                  'causality_established': False,
                  'note': 'Synthetic model outputs support demo sampling review only; they do not establish a process cause or authorize machine changes.'}
    if fraction is not None:
        supporting['predicted_defect_fraction'] = fraction
        interval = uncertainty.get('interval_95')
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            low, high = map(_fraction, interval)
            if low is not None and high is not None and low <= high:
                supporting['simulated_predictive_interval_95'] = [low, high]
        breach = _fraction(uncertainty.get('breach_probability'))
        if breach is not None:
            supporting['simulated_tolerance_breach_probability'] = breach
        threshold = _fraction(uncertainty.get('tolerance'))
        threshold = .5 if threshold is None else threshold
        if fraction >= threshold:
            steps.append('Review increased next-lot sampling with the engineer because the synthetic forecast meets or exceeds the demo sampling threshold.')
    return {'required': bool(defects), 'text': ' '.join(steps),
            'status': 'pending' if defects else 'not_required',
            'execution': 'Engineer-controlled; no PLC connection', 'sop_version': SOP_VERSION,
            'steps': steps, 'containment': 'affected_lot' if critical else ('affected_part' if defects else 'sampling_only'),
            'hypothesis': hypothesis if isinstance(hypothesis, str) else None,
            'hypothesis_note': 'Ranked non-causal hypothesis, not a diagnosed root cause.',
            'supporting_evidence': supporting}
