from copy import deepcopy
from threading import RLock
import hashlib
import json
from sqlalchemy import create_engine, select, update, event
from sqlalchemy.orm import Session
from core.config import DATABASE_URL, ROOT
from database.models import (Base, Inspection, Part, Telemetry, RiskAndRCA, CorrectiveAction,
    InspectionRevision, AuditEvent, AnalyticsJob, Frame, ModelRun, SensorDataset, HistoricalSensorReading)

class ConflictError(Exception): pass

def event_hash(previous, payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256((previous + canonical).encode()).hexdigest()

def evidence_digest(payload):
    fields = ['id', 'created_at', 'part_id', 'lot_id', 'machine_id', 'source', 'image_sha256',
              'defects', 'telemetry', 'quality', 'calibration', 'scenario', 'seed', 'model_versions']
    return event_hash('', {key: payload[key] for key in fields})

class Repository:
    def __init__(self, url=DATABASE_URL):
        (ROOT / 'data').mkdir(exist_ok=True)
        self.engine = create_engine(url, connect_args={'check_same_thread': False, 'timeout': 15} if url.startswith('sqlite') else {'connect_timeout': 10}, pool_pre_ping=True, pool_recycle=300, pool_size=5, max_overflow=5)
        if url.startswith('sqlite'):
            @event.listens_for(self.engine, 'connect')
            def configure_sqlite(connection, _):
                connection.execute('PRAGMA foreign_keys=ON')
                connection.execute('PRAGMA journal_mode=WAL')
        Base.metadata.create_all(self.engine)
        self.lock = RLock()
        self._backfill()

    def _backfill(self):
        # Additive upgrade from the initial JSON-only local scaffold. No evidence deleted.
        with self.lock, Session(self.engine) as session, session.begin():
            for record in session.scalars(select(Inspection)):
                if session.get(InspectionRevision, record.id) is not None: continue
                payload = deepcopy(record.payload)
                payload['audit'][0]['evidence_digest'] = evidence_digest(payload)
                record.payload = payload
                if session.get(Part, payload['part_id']) is None:
                    session.add(Part(id=payload['part_id'], lot_id=payload['lot_id'], machine_id=payload['machine_id'], source=payload['source']))
                    session.flush()
                session.add(Telemetry(inspection_id=record.id, part_id=payload['part_id'], observed_at=record.created_at, values=payload['telemetry']))
                session.add(InspectionRevision(inspection_id=record.id, version=1))
                self._projections(session, payload); self._audit(session, payload)

    def _audit(self, session, payload, start=0):
        previous = session.get(AuditEvent, f"{payload['id']}:{start-1}") if start else None
        digest = previous.event_hash if previous else '0' * 64
        for sequence, item in enumerate(payload['audit'][start:], start):
            new_hash = event_hash(digest, item)
            session.add(AuditEvent(id=f"{payload['id']}:{sequence}", inspection_id=payload['id'],
                sequence=sequence, event=item, previous_hash=digest, event_hash=new_hash))
            digest = new_hash

    def _projections(self, session, payload):
        session.merge(CorrectiveAction(inspection_id=payload['id'], status=payload['action']['status'], payload=payload['action']))
        if payload.get('analytics') is not None:
            session.merge(RiskAndRCA(inspection_id=payload['id'], payload=payload['analytics']))

    def save(self, payload, job=None):
        payload['audit'][0]['evidence_digest'] = evidence_digest(payload)
        with self.lock, Session(self.engine) as session, session.begin():
            part = session.get(Part, payload['part_id'])
            if part is None:
                session.add(Part(id=payload['part_id'], lot_id=payload['lot_id'], machine_id=payload['machine_id'], source=payload['source']))
                session.flush()
            elif part.lot_id != payload['lot_id'] or part.machine_id != payload['machine_id']:
                raise ValueError('Part traceability cannot change across inspections')
            session.add(Inspection(id=payload['id'], created_at=payload['created_at'], payload=payload))
            session.flush()
            session.add(InspectionRevision(inspection_id=payload['id'], version=1))
            session.add(Telemetry(inspection_id=payload['id'], part_id=payload['part_id'], observed_at=payload['created_at'], values=payload['telemetry']))
            self._projections(session, payload)
            self._audit(session, payload)
            if job: session.add(AnalyticsJob(**job))
        return payload

    def get(self, identifier):
        with Session(self.engine) as session:
            record = session.get(Inspection, identifier)
            return deepcopy(record.payload) if record else None

    def list(self, limit=100, offset=0, lot_id=None, machine_id=None):
        with Session(self.engine) as session:
            query = select(Inspection)
            if lot_id or machine_id:
                query = query.join(Telemetry, Telemetry.inspection_id == Inspection.id).join(Part, Part.id == Telemetry.part_id)
                if lot_id: query = query.where(Part.lot_id == lot_id)
                if machine_id: query = query.where(Part.machine_id == machine_id)
            return [deepcopy(r.payload) for r in session.scalars(query.order_by(Inspection.created_at.desc(), Inspection.id).offset(offset).limit(limit))]

    def update(self, identifier, transform):
        with self.lock, Session(self.engine) as session, session.begin():
            record = session.get(Inspection, identifier)
            if record is None: raise KeyError(identifier)
            revision = session.get(InspectionRevision, identifier)
            if revision is None:
                revision = InspectionRevision(inspection_id=identifier, version=1)
                session.add(revision); session.flush()
                self._audit(session, record.payload)
            old = deepcopy(record.payload)
            payload = transform(deepcopy(old))
            if payload['audit'][:len(old['audit'])] != old['audit']:
                raise ValueError('Existing audit events are immutable')
            for field in ['id', 'created_at', 'part_id', 'lot_id', 'machine_id', 'source', 'image_sha256', 'defects', 'telemetry', 'quality', 'calibration', 'scenario', 'seed', 'model_versions']:
                if payload[field] != old[field]: raise ValueError(f'Inspection evidence is immutable: {field}')
            changed = session.execute(update(InspectionRevision).where(InspectionRevision.inspection_id == identifier,
                InspectionRevision.version == revision.version).values(version=revision.version+1), execution_options={'synchronize_session': False})
            if changed.rowcount != 1: raise ConflictError('Concurrent update detected; reload and retry')
            record.payload = payload
            self._projections(session, payload)
            self._audit(session, payload, len(old['audit']))
            return payload

    def audit(self, identifier):
        with Session(self.engine) as session:
            records = session.scalars(select(AuditEvent).where(AuditEvent.inspection_id == identifier).order_by(AuditEvent.sequence))
            return [{'sequence': r.sequence, 'event': r.event, 'previous_hash': r.previous_hash, 'event_hash': r.event_hash} for r in records]

    def verify_audit(self, identifier):
        snapshot = self.get(identifier)
        if snapshot is None: raise KeyError(identifier)
        records = self.audit(identifier); previous = '0' * 64
        valid = len(records) == len(snapshot['audit'])
        for i, record in enumerate(records):
            valid = valid and record['sequence'] == i and record['previous_hash'] == previous and event_hash(previous, record['event']) == record['event_hash']
            previous = record['event_hash']
        valid = valid and [r['event'] for r in records] == snapshot['audit']
        valid = valid and bool(snapshot['audit']) and snapshot['audit'][0].get('evidence_digest') == evidence_digest(snapshot)
        return {'valid': valid, 'events': len(records), 'head_hash': previous,
                'limitation': 'Detects inconsistency; not a signed ledger against a privileged database editor.'}

    def save_frame(self, payload):
        with Session(self.engine) as session, session.begin(): session.add(Frame(**payload))

    def get_frame(self, identifier):
        with Session(self.engine) as session:
            r = session.get(Frame, identifier)
            if r is None: return None
            return {'id': r.id, 'created_at': r.created_at, 'sha256': r.sha256,
                    'media_type': r.media_type, 'content': r.content, 'quality': deepcopy(r.quality)}

    def get_job(self, identifier):
        with Session(self.engine) as session:
            record = session.get(AnalyticsJob, identifier)
            return {k: getattr(record, k) for k in ['id', 'inspection_id', 'status', 'created_at', 'error']} if record else None

    def claim_job(self):
        with self.lock, Session(self.engine) as session, session.begin():
            candidate = session.scalar(select(AnalyticsJob).where(AnalyticsJob.status == 'queued').order_by(AnalyticsJob.created_at).limit(1))
            if not candidate: return None
            result = session.execute(update(AnalyticsJob).where(AnalyticsJob.id == candidate.id, AnalyticsJob.status == 'queued').values(status='running'), execution_options={'synchronize_session': False})
            return candidate.id if result.rowcount == 1 else None

    def finish_job(self, identifier, status, error=None):
        with Session(self.engine) as session, session.begin():
            session.execute(update(AnalyticsJob).where(AnalyticsJob.id == identifier, AnalyticsJob.status == 'running').values(status=status, error=error))

    def recover_jobs(self):
        with Session(self.engine) as session, session.begin():
            session.execute(update(AnalyticsJob).where(AnalyticsJob.status == 'running').values(status='queued', error=None))

    def retry_job(self, identifier):
        with Session(self.engine) as session, session.begin():
            result = session.execute(update(AnalyticsJob).where(AnalyticsJob.id == identifier,
                AnalyticsJob.status == 'failed').values(status='queued', error=None))
            if result.rowcount != 1: raise ConflictError('Only failed jobs can be retried')

    def save_model_run(self, payload):
        with Session(self.engine) as session, session.begin():
            session.add(ModelRun(id=payload['id'], frame_id=payload['frame_id'], created_at=payload['created_at'], payload=payload))

    def get_model_run(self, identifier):
        with Session(self.engine) as session:
            record = session.get(ModelRun, identifier)
            return deepcopy(record.payload) if record else None

    def import_sensor_dataset(self, dataset, readings):
        """Atomically save an import manifest and its deduplicated sensor observations."""
        from sqlalchemy import select
        with self.lock, Session(self.engine) as session, session.begin():
            existing_dataset = session.get(SensorDataset, dataset['sha256'])
            if existing_dataset:
                return {'sha256': existing_dataset.sha256, 'filename': existing_dataset.filename,
                    'source_label': existing_dataset.source_label, 'rows_read': existing_dataset.rows_read,
                    'rows_inserted': 0, 'rows_duplicate': existing_dataset.rows_read, 'already_imported': True}
            candidates = {item['id']: item for item in readings}
            present = set(session.scalars(select(HistoricalSensorReading.id).where(HistoricalSensorReading.id.in_(candidates)))) if candidates else set()
            new_readings = [item for key, item in candidates.items() if key not in present]
            manifest = SensorDataset(**(dataset | {'rows_inserted': len(new_readings), 'rows_duplicate': dataset['rows_duplicate'] + len(readings) - len(new_readings)}))
            session.add(manifest)
            session.add_all(HistoricalSensorReading(**item) for item in new_readings)
            return {'sha256': manifest.sha256, 'filename': manifest.filename, 'source_label': manifest.source_label,
                'rows_read': manifest.rows_read, 'rows_inserted': manifest.rows_inserted,
                'rows_duplicate': manifest.rows_duplicate, 'already_imported': False}

    def sensor_catalog(self):
        from sqlalchemy import select, func
        with Session(self.engine) as session:
            machines = list(session.scalars(select(HistoricalSensorReading.machine_id).distinct().order_by(HistoricalSensorReading.machine_id)))
            sensors = list(session.scalars(select(HistoricalSensorReading.sensor_name).distinct().order_by(HistoricalSensorReading.sensor_name)))
            return {'machines': machines, 'sensors': sensors, 'reading_count': session.scalar(select(func.count()).select_from(HistoricalSensorReading)) or 0,
                'dataset_count': session.scalar(select(func.count()).select_from(SensorDataset)) or 0,
                'source': 'imported historical sensor readings; source labels are user supplied'}

    def sensor_history(self, machine_id, sensor_name, start, end, limit=500):
        from sqlalchemy import select, and_, func
        from datetime import timedelta
        with Session(self.engine) as session:
            where = and_(HistoricalSensorReading.machine_id == machine_id,
                HistoricalSensorReading.sensor_name == sensor_name, HistoricalSensorReading.observed_at >= start,
                HistoricalSensorReading.observed_at < end)
            count = session.scalar(select(func.count()).select_from(HistoricalSensorReading).where(where)) or 0
            # Uniformly thin very high-rate telemetry in SQL before returning chart points.
            stride = max(1, count // limit)
            ranked = select(HistoricalSensorReading.id.label('id'),
                func.row_number().over(order_by=HistoricalSensorReading.observed_at).label('rn')).where(where).subquery()
            sampled_ids = select(ranked.c.id).where((ranked.c.rn % stride) == 0).limit(limit)
            rows = list(session.scalars(select(HistoricalSensorReading).where(HistoricalSensorReading.id.in_(sampled_ids)).order_by(HistoricalSensorReading.observed_at)))
            aggregates = [func.avg(HistoricalSensorReading.value), func.min(HistoricalSensorReading.value),
                          func.max(HistoricalSensorReading.value)]
            if self.engine.dialect.name == 'postgresql':
                aggregates.append(func.stddev_samp(HistoricalSensorReading.value))
            stats = session.execute(select(*aggregates).where(where)).one()
            if self.engine.dialect.name == 'sqlite':
                import statistics
                values = list(session.scalars(select(HistoricalSensorReading.value).where(where)))
                stddev = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                stddev = stats[3] or 0.0
            if count and rows:
                last = session.execute(select(HistoricalSensorReading).where(where).order_by(HistoricalSensorReading.observed_at.desc()).limit(1)).scalar_one()
                if rows[-1].id != last.id:
                    rows = (rows + [last])[-limit:]
            points = [{'observed_at': item.observed_at.isoformat(), 'value': item.value, 'unit': item.unit, 'lot_id': item.lot_id} for item in rows]
            return {'points': points, 'count': count, 'mean': stats[0], 'min': stats[1], 'max': stats[2],
                'stddev': stddev, 'unit': points[0]['unit'] if points else None}
