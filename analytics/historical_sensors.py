"""Validated, provenance-labelled CSV import for historical machine sensor readings."""
import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import PurePath

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 50_000
REQUIRED = {'timestamp', 'machine_id', 'sensor', 'value', 'unit'}
IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,79}$')

class SensorCSVError(ValueError):
    pass

def parse_csv(content: bytes, filename: str, source_label: str):
    if not content:
        raise SensorCSVError('The CSV is empty.')
    if len(content) > MAX_BYTES:
        raise SensorCSVError('CSV exceeds the 5 MiB upload limit.')
    try:
        stream = io.StringIO(content.decode('utf-8-sig'), newline='')
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not REQUIRED.issubset(set(fields)):
            raise SensorCSVError('CSV needs unique headers: timestamp, machine_id, sensor, value, unit. lot_id is optional.')
        readings = []
        seen = set()
        rows_read = 0
        local_duplicates = 0
        for line, row in enumerate(reader, start=2):
            if line > MAX_ROWS + 1:
                raise SensorCSVError(f'CSV exceeds the {MAX_ROWS:,} row import limit.')
            if None in row or any(value is None for value in row.values()):
                raise SensorCSVError(f'Row {line} has missing or extra columns.')
            rows_read += 1
            observed = datetime.fromisoformat(row['timestamp'].strip().replace('Z', '+00:00'))
            if observed.tzinfo is None or observed.utcoffset() is None:
                raise SensorCSVError(f'Row {line} timestamp needs a timezone, such as 2026-04-15T10:00:00Z.')
            observed = observed.astimezone(timezone.utc)
            machine = row['machine_id'].strip()
            sensor = row['sensor'].strip().lower()
            unit = row['unit'].strip()
            lot = (row.get('lot_id') or '').strip() or None
            if not IDENTIFIER.fullmatch(machine) or not IDENTIFIER.fullmatch(sensor) or not unit or len(unit) > 32:
                raise SensorCSVError(f'Row {line} has an invalid machine, sensor, or unit value.')
            if unit.strip().lower() in ('nan', 'none', 'null'):
                raise SensorCSVError(f'Row {line} unit cannot be empty.')
            if lot and not IDENTIFIER.fullmatch(lot):
                raise SensorCSVError(f'Row {line} has an invalid lot_id.')
            try:
                value = float(row['value'])
            except ValueError:
                raise SensorCSVError(f'Row {line} value must be numeric.') from None
            if not math.isfinite(value) or abs(value) > 1e15:
                raise SensorCSVError(f'Row {line} value must be finite and within supported range.')
            canonical = [machine, sensor, observed.isoformat(), value, unit]
            observation_id = hashlib.sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest()
            if observation_id in seen:
                local_duplicates += 1
                continue
            seen.add(observation_id)
            readings.append({'id': observation_id, 'machine_id': machine, 'sensor_name': sensor,
                'observed_at': observed, 'value': value, 'unit': unit, 'lot_id': lot})
    except (UnicodeDecodeError, csv.Error, ValueError) as error:
        if isinstance(error, SensorCSVError):
            raise
        raise SensorCSVError('CSV format or timestamp is invalid. Use UTF-8 and ISO-8601 timestamps with time zones.') from None
    if not readings:
        raise SensorCSVError('CSV contains a header but no sensor readings.')
    sha = hashlib.sha256(content).hexdigest()
    safe_name = PurePath(filename.replace('\\', '/')).name[:180] or 'sensor-history.csv'
    label = source_label.strip()[:180] or 'source not verified'
    dataset = {'sha256': sha, 'filename': safe_name, 'source_label': label,
        'imported_at': datetime.now(timezone.utc), 'rows_read': rows_read, 'rows_duplicate': local_duplicates}
    return dataset, readings
