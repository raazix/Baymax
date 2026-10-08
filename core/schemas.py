from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class DefectClass(StrEnum):
    CRACK = 'surface_crack'
    SCRATCH = 'scratch'
    POROSITY = 'porosity'
    BURR = 'burr'
    DENT = 'dent'
    CORROSION = 'corrosion'
    DEFORMATION = 'deformation'
    UNKNOWN = 'anomaly_unclassified'

class Detection(StrictModel):
    label: DefectClass
    confidence: float = Field(ge=0, le=1)
    centroid_px: tuple[float, float]
    area_px: float = Field(gt=0)
    length_px: float = Field(gt=0)
    polygon_px: list[tuple[float, float]] = Field(min_length=3)

class ReplayRequest(StrictModel):
    scenario: Literal['thermal_drift', 'cosmetic_scratch', 'normal', 'unknown_anomaly', 'blurred'] = 'thermal_drift'
    seed: int = Field(default=42, ge=0, le=2147483647)
    analytics_mode: Literal['inline', 'deferred'] = 'inline'

class FrameResponse(StrictModel):
    id: str
    created_at: str
    sha256: str
    media_type: str
    quality: dict
    image_url: str
    source: Literal['uploaded_frame'] = 'uploaded_frame'

class JobResponse(StrictModel):
    id: str
    inspection_id: str
    status: Literal['queued', 'running', 'completed', 'failed']
    created_at: str
    error: str | None

class InspectionResponse(StrictModel):
    id: str
    created_at: str
    source: Literal['synthetic_replay', 'uploaded_image']
    scenario: str
    seed: int
    part_id: str
    lot_id: str
    machine_id: str
    quality: dict
    calibration: dict
    defects: list[dict]
    telemetry: dict[str, float]
    spatial_fingerprint: dict
    disposition: Literal['review', 'pass', 'recapture']
    image_url: str
    image_sha256: str
    model_versions: dict[str, str]
    audit: list[dict]
    analytics: dict | None
    action: dict
    analytics_job_id: str | None = None
    verification: dict | None = None
    context: dict | None = None

class DecisionRequest(StrictModel):
    decision: Literal['approve', 'reject', 'escalate']
    engineer: str = Field(min_length=2, max_length=100, pattern=r'.*\S.*')
    note: str = Field(default='', max_length=1000)

class VerificationRequest(StrictModel):
    inspection_ids: list[str] = Field(min_length=20, max_length=1000)


class WhatIfRequest(StrictModel):
    temperature_c: float = Field(ge=400, le=1000)
    pressure_bar: float = Field(ge=0, le=300)
    vibration_mm_s: float = Field(ge=0, le=50)
    machine_speed_rpm: float = Field(ge=0, le=5000)
