from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import JSON, String, Integer, ForeignKey, LargeBinary, Float, DateTime, Index, UniqueConstraint

class Base(DeclarativeBase): pass

class Inspection(Base):
    __tablename__ = 'inspections'
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON)

class Part(Base):
    __tablename__ = 'parts'
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    lot_id: Mapped[str] = mapped_column(String(100), index=True)
    machine_id: Mapped[str] = mapped_column(String(100), index=True)
    source: Mapped[str] = mapped_column(String(40))

class Telemetry(Base):
    __tablename__ = 'process_telemetry'
    inspection_id: Mapped[str] = mapped_column(ForeignKey('inspections.id'), primary_key=True)
    part_id: Mapped[str] = mapped_column(ForeignKey('parts.id'), index=True)
    observed_at: Mapped[str] = mapped_column(String(40))
    values: Mapped[dict] = mapped_column(JSON)

class RiskAndRCA(Base):
    __tablename__ = 'risk_and_rca'
    inspection_id: Mapped[str] = mapped_column(ForeignKey('inspections.id'), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)

class CorrectiveAction(Base):
    __tablename__ = 'corrective_actions'
    inspection_id: Mapped[str] = mapped_column(ForeignKey('inspections.id'), primary_key=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    payload: Mapped[dict] = mapped_column(JSON)

class InspectionRevision(Base):
    __tablename__ = 'inspection_revisions'
    inspection_id: Mapped[str] = mapped_column(ForeignKey('inspections.id'), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1)

class AuditEvent(Base):
    __tablename__ = 'audit_events'
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey('inspections.id'), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    event: Mapped[dict] = mapped_column(JSON)
    previous_hash: Mapped[str] = mapped_column(String(64))
    event_hash: Mapped[str] = mapped_column(String(64))

class AnalyticsJob(Base):
    __tablename__ = 'analytics_jobs'
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey('inspections.id'), unique=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[str] = mapped_column(String(40))
    error: Mapped[str | None] = mapped_column(String(300), nullable=True)

class Frame(Base):
    __tablename__ = 'frames'
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    media_type: Mapped[str] = mapped_column(String(40))
    content: Mapped[bytes] = mapped_column(LargeBinary)
    quality: Mapped[dict] = mapped_column(JSON)

class ModelRun(Base):
    __tablename__ = 'model_runs'
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    frame_id: Mapped[str] = mapped_column(ForeignKey('frames.id'), index=True)
    created_at: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON)

class SensorDataset(Base):
    __tablename__ = 'sensor_datasets'
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(180))
    source_label: Mapped[str] = mapped_column(String(180), default='source not verified')
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    rows_read: Mapped[int] = mapped_column(Integer)
    rows_inserted: Mapped[int] = mapped_column(Integer)
    rows_duplicate: Mapped[int] = mapped_column(Integer)

class HistoricalSensorReading(Base):
    __tablename__ = 'historical_sensor_readings'
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_sha256: Mapped[str] = mapped_column(ForeignKey('sensor_datasets.sha256'), index=True)
    machine_id: Mapped[str] = mapped_column(String(80))
    sensor_name: Mapped[str] = mapped_column(String(80))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32))
    lot_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    __table_args__ = (
        UniqueConstraint('machine_id', 'sensor_name', 'observed_at', 'value', 'unit', name='uq_sensor_observation'),
        Index('ix_sensor_machine_name_time', 'machine_id', 'sensor_name', 'observed_at'),
    )
