"""Turn imported historical sensor readings into lot-level process context for an inspection.

The current lot's sensor means become the inspection telemetry (RCA input); the two preceding lots become the PLSR
forecast history, replacing the "repeat current telemetry" persistence assumption. Drift flags compare the current lot
with the machine's earlier baseline lots. Provenance is the import source label, which says SYNTHETIC for the
generated demo history, and is carried into the inspection record.
"""
import math

FEATURES = ('temperature_c', 'pressure_bar', 'vibration_mm_s', 'machine_speed_rpm')
BASELINE_EXCLUDE = 14      # the most recent 14 lots (7 days of 12-hour lots) are not used as baseline
MIN_BASELINE = 8
DRIFT_Z = 3.0


def _complete(lot):
    return all(name in lot['values'] and math.isfinite(lot['values'][name]) for name in FEATURES)


def build_process_context(aggregates: dict, machine_id: str) -> dict | None:
    """Return telemetry, forecast history, trend and drift flags for the machine's latest complete lot, or None."""
    lots = [lot for lot in aggregates.get('lots', []) if _complete(lot)]
    if not lots:
        return None
    current, prior = lots[-1], lots[-3:-1]
    baseline = lots[:-BASELINE_EXCLUDE] if len(lots) - BASELINE_EXCLUDE >= MIN_BASELINE else lots[:-1]
    drift = []
    for name in FEATURES:
        values = [lot['values'][name] for lot in baseline]
        if len(values) < 3:
            continue
        mean = sum(values) / len(values)
        spread = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) or 1e-9
        z = (current['values'][name] - mean) / spread
        if abs(z) >= DRIFT_Z:
            drift.append({'sensor': name, 'current': round(current['values'][name], 3), 'baseline_mean': round(mean, 3),
                          'change': round(current['values'][name] - mean, 3), 'z_score': round(z, 1),
                          'unit': current['units'].get(name)})
    drift.sort(key=lambda item: -abs(item['z_score']))
    labels = aggregates.get('source_labels', [])
    synthetic = any('SYNTHETIC' in label.upper() for label in labels)
    return {
        'machine_id': machine_id,
        'lot_id': current['lot_id'],
        'lot_window': [current['start'], current['end']],
        'readings': current['readings'],
        'telemetry': {name: round(current['values'][name], 3) for name in FEATURES},
        'units': {name: current['units'].get(name) for name in FEATURES},
        'forecast_history': [{name: lot['values'][name] for name in FEATURES} for lot in prior],
        'prior_lots': [lot['lot_id'] for lot in prior],
        'trend': [{'lot_id': lot['lot_id'], 'end': lot['end'], **{name: round(lot['values'][name], 3) for name in FEATURES}}
                  for lot in lots[-20:]],
        'drift': drift,
        'baseline_lots': len(baseline),
        'source_labels': labels,
        'synthetic': synthetic,
        'telemetry_source': (f"{'SYNTHETIC ' if synthetic else ''}historical sensor readings for machine {machine_id}, "
                             f"lot {current['lot_id']} ({current['readings']} readings); imported source: "
                             + ('; '.join(labels)[:300] or 'unlabelled')),
    }
