"""Generate clearly labelled SYNTHETIC historical sensor/lot data for the LineGuard demo.

No real machine data exists for this project. These CSVs reproduce the process scenarios the RCA and forecast models
were trained on (see scripts/train_xgboost.py) so the pipeline can be exercised end to end with lot history. Every
file is imported with a source label that says SYNTHETIC; nothing here is a measurement.

Usage:
    .venv\\Scripts\\python.exe scripts/generate_sensor_history.py              # write CSVs to data/synthetic_history
    .venv\\Scripts\\python.exe scripts/generate_sensor_history.py --import     # also import through the running API
"""
import argparse
import csv
import io
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'synthetic_history'
SENSORS = (('temperature_c', '°C'), ('pressure_bar', 'bar'), ('vibration_mm_s', 'mm/s'), ('machine_speed_rpm', 'rpm'))
NOMINAL = np.array([705.0, 100.0, 2.0, 1200.0])
LOT_NOISE = np.array([4.0, 2.0, .2, 20.0])        # lot-to-lot variation
READING_NOISE = np.array([3.0, 1.5, .25, 15.0])   # within-lot sensor noise
DAYS, LOT_HOURS, STEP_MINUTES = 30, 12, 15

# Each machine tells one story. Offsets mirror the RCA training scenarios (temperature, pressure, vibration, speed).
MACHINES = {
    'M-01': {'name': 'Casting cell 1', 'story': 'stable nominal process', 'onset_day': None, 'offset': [0, 0, 0, 0]},
    'M-02': {'name': 'Casting cell 2', 'story': 'thermal drift developing over the last 6 days', 'onset_day': 24, 'offset': [50, 3, .3, 0]},
    'M-03': {'name': 'Machining cell 3', 'story': 'tooling vibration rising over the last 10 days', 'onset_day': 20, 'offset': [0, 2, 3.4, 30]},
    'M-04': {'name': 'Casting cell 4', 'story': 'intermittent pressure instability in recent lots', 'onset_day': 22, 'offset': [4, -22, .2, 0]},
}


def generate(machine: str, spec: dict, end: datetime, seed: int):
    rng = np.random.default_rng(seed)
    start = end - timedelta(days=DAYS)
    rows = []
    lots = int(DAYS * 24 / LOT_HOURS)
    for lot in range(lots):
        lot_start = start + timedelta(hours=lot * LOT_HOURS)
        day = lot * LOT_HOURS / 24
        level = NOMINAL + rng.normal(0, LOT_NOISE)
        onset = spec['onset_day']
        if onset is not None and day >= onset:
            progress = min(1.0, (day - onset) / (DAYS - onset))
            offset = np.array(spec['offset'], dtype=float)
            if machine == 'M-04':          # episodes, not a ramp
                offset = offset * (1.0 if rng.random() < .55 else .15)
            else:
                offset = offset * progress
            level = level + offset
        lot_id = f'{machine}-L{lot_start:%m%d}{"A" if lot_start.hour < 12 else "B"}'
        steps = LOT_HOURS * 60 // STEP_MINUTES
        for step in range(steps):
            at = lot_start + timedelta(minutes=step * STEP_MINUTES)
            values = level + rng.normal(0, READING_NOISE)
            for (sensor, unit), value in zip(SENSORS, values):
                rows.append([at.strftime('%Y-%m-%dT%H:%M:%SZ'), machine, sensor, f'{value:.3f}', unit, lot_id])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--end', help='UTC end time, ISO format; default is the start of the current hour')
    parser.add_argument('--import', dest='do_import', action='store_true', help='POST each CSV to the running API')
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    end = datetime.fromisoformat(args.end.replace('Z', '+00:00')) if args.end else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'generated_at': datetime.now(timezone.utc).isoformat(), 'end': end.isoformat(), 'seed': args.seed,
                'synthetic': True, 'note': 'Generated demo data. Not machine measurements.', 'machines': {}}
    for index, (machine, spec) in enumerate(MACHINES.items()):
        rows = generate(machine, spec, end, args.seed + index)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(['timestamp', 'machine_id', 'sensor', 'value', 'unit', 'lot_id'])
        writer.writerows(rows)
        path = OUT / f'{machine}_synthetic_history.csv'
        path.write_text(buffer.getvalue(), encoding='utf-8')
        entry = {'file': path.name, 'rows': len(rows), 'name': spec['name'], 'story': spec['story']}
        if args.do_import:
            label = f"SYNTHETIC demo history - {spec['name']}: {spec['story']} (seed {args.seed + index})"
            query = urllib.parse.urlencode({'filename': path.name, 'source_label': label[:180]})
            request = urllib.request.Request(f'{args.api}/api/history/sensor-datasets?{query}', data=path.read_bytes(),
                                             method='POST', headers={'Content-Type': 'text/csv'})
            with urllib.request.urlopen(request, timeout=180) as response:
                entry['import'] = json.load(response)
        manifest['machines'][machine] = entry
        print(machine, entry['rows'], 'rows', entry.get('import', {}).get('rows_inserted', 'written'))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
