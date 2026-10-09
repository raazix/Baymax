"""Rank machines and their batches by predicted next-lot defect risk from production history.

For every machine with imported sensor history: the latest lot's sensor means (with its two prior lots) go through the
same trained analytics as an inspection - XGBoost root-cause hypothesis with TreeSHAP drivers, PLSR next-lot defect
fraction and the Monte Carlo predictive range. The same scoring is repeated for each of the last batches to show the
trend. Observed inspection results for the machine are reported alongside as evidence, not mixed into the forecast.

All models are trained on synthetic process data and the demo history is synthetic; confidence values are the
uncalibrated classifier probability. Rankings support engineer prioritisation, not automatic action.
"""
from actions.sop_engine import HYPOTHESIS_ACTIONS
from analytics.lot_history import FEATURES, build_process_context
from analytics.service import analyze

SEVERITY_ORDER = ('low', 'medium', 'review_required', 'high', 'critical')
BATCH_TREND = 6


def _score(context, seed):
    result = analyze(context['telemetry'] | {'history': context['forecast_history']}, seed)
    rca, forecast, uncertainty = result['rca'], result['forecast'], result['uncertainty']
    return {'predicted_defect_fraction': round(forecast['risk'], 4), 'interval_95': [round(v, 4) for v in uncertainty['interval_95']],
            'probability_over_tolerance': round(uncertainty['breach_probability'], 4), 'tolerance': uncertainty['tolerance'],
            'root_cause': rca['hypothesis'], 'root_cause_confidence': round(rca['confidence'], 4) if rca.get('confidence') is not None else None,
            'top_drivers': [{'feature': c['feature'], 'shap_margin': round(c['contribution'], 3)} for c in rca['feature_contributions'][:3]],
            'history_assumption': forecast['history_assumption']}


def _action(score):
    cause = score['root_cause']
    steps = []
    if score['probability_over_tolerance'] >= .05 or score['predicted_defect_fraction'] >= .35:
        steps.append('Increase sampling for the next lot and hold it for engineer review before dispatch.')
    if cause in HYPOTHESIS_ACTIONS:
        steps.append(HYPOTHESIS_ACTIONS[cause])
    elif not steps:
        steps.append('No process cause indicated; continue normal sampling and monitor the trend.')
    else:
        steps.append('No single process cause indicated; review the drifting sensors with the process engineer.')
    return ' '.join(steps)


def _observed(records):
    passed = [r for r in records if (r.get('quality') or {}).get('passed')]
    defective = [r for r in passed if r.get('defects')]
    worst = None
    for r in defective:
        for d in r['defects']:
            level = (d.get('severity') or {}).get('level')
            if level in SEVERITY_ORDER and (worst is None or SEVERITY_ORDER.index(level) > SEVERITY_ORDER.index(worst)):
                worst = level
    return {'inspected': len(passed), 'with_findings': len(defective), 'worst_severity': worst,
            'last_inspected_at': records[0]['created_at'] if records else None}


def rank_machines(repo, seed=2026):
    machines = repo.sensor_catalog()['machines']
    ranking = []
    for machine in machines:
        aggregates = repo.lot_aggregates(machine, 60)
        context = build_process_context(aggregates, machine)
        if context is None:
            continue
        score = _score(context, seed)
        lots = [lot for lot in aggregates['lots'] if all(name in lot['values'] for name in FEATURES)]
        batches = []
        for end in range(max(3, len(lots) - BATCH_TREND + 1), len(lots) + 1):
            batch_context = build_process_context({'lots': lots[:end], 'source_labels': aggregates['source_labels']}, machine)
            batch_score = _score(batch_context, seed)
            batches.append({'lot_id': batch_context['lot_id'], 'produced': batch_context['lot_window'],
                            'predicted_defect_fraction_next': batch_score['predicted_defect_fraction'],
                            'root_cause': batch_score['root_cause'], 'root_cause_confidence': batch_score['root_cause_confidence']})
        ranking.append({'machine_id': machine, 'current_lot': context['lot_id'], 'produced': context['lot_window'],
                        'readings': context['readings'], 'process': context['telemetry'], 'units': context['units'],
                        'drift': context['drift'], **score, 'recommended_action': _action(score), 'batches': batches,
                        'observed_inspections': _observed(repo.list(limit=50, machine_id=machine)),
                        'synthetic_history': context['synthetic']})
    ranking.sort(key=lambda item: (item['predicted_defect_fraction'], item['probability_over_tolerance']), reverse=True)
    for index, item in enumerate(ranking, 1):
        item['rank'] = index
    return {'ranking': ranking, 'seed': seed,
            'method': 'Latest lot per machine scored with trained XGBoost + TreeSHAP (root cause), PLSR (next-lot defect fraction) '
                      'and a 10,000-draw Monte Carlo range; ranked by predicted defect fraction, then probability over tolerance.',
            'limitations': 'Models trained on synthetic process data; demo history is synthetic; root-cause confidence is an uncalibrated '
                           'classifier probability and not causal proof. Engineer approval is required for any action.'}
