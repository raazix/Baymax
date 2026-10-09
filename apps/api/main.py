import hashlib
import json
import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import secrets
from typing import Literal
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import Response, JSONResponse
from starlette.concurrency import run_in_threadpool
from core.schemas import WhatIfRequest, ReplayRequest, DecisionRequest, VerificationRequest, FrameResponse, JobResponse, InspectionResponse
from core.replay import create_replay, render_svg
from core.upload_inspection import create_upload_inspection, PROFILES
from core.part_reference import identify_reference
from database.repository import Repository, ConflictError
from core.clock import utc_now
from vision.image_ingestion import decode, MAX_BYTES
from vision.quality_gate import check_frame, check_frame_open_domain, profile_status, QualityCalibrationUnavailable
from vision.yolo_detector import ProxyDetector, ModelUnavailable
from vision.patchcore_anomaly import PatchCoreDetector, PatchCoreUnavailable
from vision import patchcore_custom
from analytics.briefing import evidence_packet
from analytics import assistant
from analytics import visual_assistant
from analytics import historical_sensors
from analytics.lot_history import build_process_context
from analytics import supermemory
from pydantic import BaseModel, ConfigDict, Field
from analytics.service import analyze as analyze_process, status as analytics_status, model_versions as analytics_versions, AnalyticsUnavailable

app = FastAPI(title='LineGuard', version='0.1.0', description='Evidence-linked Track 3 hackathon scaffold')
logger = logging.getLogger(__name__)

class AssistantQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(default='Explain this inspection and the recommended next action.', min_length=1, max_length=1200)
    purpose: Literal['question', 'summary'] = 'question'

@app.exception_handler(assistant.AssistantUnavailable)
async def assistant_error(request, error):
    return JSONResponse({'detail': str(error)}, status_code=error.status)

@app.get('/api/assistant/status')
def assistant_status():
    return assistant.status() | {'supermemory': supermemory.status(), 'vision_model': os.getenv('NVIDIA_VISION_MODEL', visual_assistant.DEFAULT_MODEL)}

@app.post('/api/inspections/{identifier}/vision-description')
async def inspection_vision_description(identifier: str):
    record = await run_in_threadpool(inspection, identifier)
    frame_id = (record.get('context') or {}).get('frame_id')
    frame = await run_in_threadpool(repo.get_frame, frame_id) if frame_id else None
    return await visual_assistant.describe(record, frame)

@app.post('/api/inspections/{identifier}/alert-speech')
async def inspection_alert_speech(identifier: str):
    record = await run_in_threadpool(inspection, identifier)
    audio = await assistant.alert_speech(record)
    return Response(audio, media_type='audio/mpeg', headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})

@app.post('/api/inspections/{identifier}/assistant')
async def inspection_assistant(identifier: str, question: AssistantQuestion):
    record = await run_in_threadpool(inspection, identifier)
    return await assistant.explain(record, question.question.strip(), purpose=question.purpose)

@app.get('/api/assistant/responses/{identifier}')
def assistant_response(identifier: str):
    return assistant.public_record(assistant.read_record(identifier))

@app.post('/api/assistant/responses/{identifier}/speech')
async def assistant_speech(identifier: str):
    record = await run_in_threadpool(assistant.read_record, identifier)
    audio = await assistant.speech(record)
    return Response(audio, media_type='audio/mpeg', headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})

@app.post('/api/assistant/transcribe')
async def assistant_transcribe(request: Request):
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > assistant.MAX_AUDIO_BYTES:
            raise HTTPException(413, 'Audio exceeds 10 MiB limit')
        content.extend(chunk)
    media_type = request.headers.get('content-type', '').split(';', 1)[0].lower()
    return await assistant.transcribe(bytes(content), media_type)
repo = Repository()
proxy_detector = ProxyDetector()
patchcore_detector = PatchCoreDetector()
from vision.rust_classifier import RustClassifier, RustUnavailable
rust_detector = RustClassifier()
custom_detectors: dict[str, PatchCoreDetector] = {}
MPDD_CATEGORIES = ('bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes')
MPDD_ROOT = Path(__file__).resolve().parents[2] / 'models/patchcore/mpdd'

def patchcore_for(name: str) -> PatchCoreDetector:
    if name == 'default': return patchcore_detector
    expected_digest = None
    if name.startswith('mpdd_'):
        category = name.removeprefix('mpdd_')
        if category not in MPDD_CATEGORIES: raise HTTPException(422, 'Unknown MPDD component category')
        path = MPDD_ROOT / category / 'casting_proxy.pt'
        report_path = path.parent / 'evaluation.json'
        if not report_path.is_file(): raise HTTPException(404, 'MPDD evaluation unavailable')
        expected_digest = json.loads(report_path.read_text())['artifact_sha256']
    else:
        try: path = patchcore_custom.artifact_path(name)
        except ValueError as error: raise HTTPException(422, str(error))
    if not path.is_file(): raise HTTPException(404, f'No trained PatchCore model named {name}')
    if name not in custom_detectors:
        custom_detectors[name] = PatchCoreDetector(artifact_path=path)
        custom_detectors[name].expected_digest = expected_digest
    return custom_detectors[name]

@app.middleware('http')
async def request_context(request: Request, call_next):
    request_id = str(uuid4())
    token = os.getenv('LINEGUARD_API_TOKEN')
    if token and request.url.path.startswith('/api/') and request.url.path != '/api/health':
        supplied = request.headers.get('authorization', '')
        if not secrets.compare_digest(supplied, 'Bearer ' + token):
            return JSONResponse({'detail': 'API authentication required', 'request_id': request_id}, status_code=401, headers={'WWW-Authenticate': 'Bearer'})
    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id
    return response

@app.exception_handler(ConflictError)
async def conflict_handler(request, error):
    return JSONResponse({'detail': str(error)}, status_code=409)

@app.exception_handler(AnalyticsUnavailable)
async def analytics_unavailable_handler(request, error):
    return JSONResponse({'detail': str(error)}, status_code=503)

@app.exception_handler(QualityCalibrationUnavailable)
async def quality_unavailable_handler(request, error):
    return JSONResponse({'detail': str(error)}, status_code=503)

async def uploaded_frame(request: Request, quality_profile='camera'):
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > MAX_BYTES: raise HTTPException(413, 'Frame exceeds 8 MiB limit')
        chunks.extend(chunk)
    content = bytes(chunks)
    try: frame, media = await run_in_threadpool(decode, content)
    except OverflowError as error: raise HTTPException(413, str(error))
    except ValueError as error: raise HTTPException(422, str(error))
    try: quality = await run_in_threadpool(check_frame_open_domain, frame, quality_profile)
    except ValueError as error: raise HTTPException(422, str(error))
    return content, frame, media, quality

@app.post('/api/camera/quality')
async def camera_quality(request: Request):
    _, _, _, quality = await uploaded_frame(request)
    return {**quality, 'source': 'camera_frame', 'defect_inference': 'not_configured'}

@app.post('/api/frames', status_code=201, response_model=FrameResponse)
async def ingest_frame(request: Request,
                       quality_profile: Literal['camera', 'neu_proxy', 'casting_proxy'] = Query('camera')):
    content, _, media, quality = await uploaded_frame(request, quality_profile)
    payload = {'id': str(uuid4()), 'created_at': utc_now(), 'sha256': hashlib.sha256(content).hexdigest(),
               'media_type': media, 'content': content, 'quality': quality}
    await run_in_threadpool(repo.save_frame, payload)
    return {k:v for k,v in payload.items() if k != 'content'} | {'image_url': f"/api/frames/{payload['id']}/image"}

@app.get('/api/frames/{identifier}', response_model=FrameResponse)
def frame_metadata(identifier: str):
    payload = repo.get_frame(identifier)
    if payload is None: raise HTTPException(404, 'Frame not found')
    return {k:v for k,v in payload.items() if k != 'content'} | {'image_url': f'/api/frames/{identifier}/image'}

@app.get('/api/frames/{identifier}/image')
def frame_image(identifier: str):
    frame = repo.get_frame(identifier)
    if frame is None: raise HTTPException(404, 'Frame not found')
    return Response(frame['content'], media_type=frame['media_type'], headers={'ETag': '"'+frame['sha256']+'"', 'X-Content-Type-Options': 'nosniff'})

@app.post('/api/proxy/frames/{identifier}/detect')
def proxy_inference(identifier: str, inference_mode: Literal['full', 'sliced'] = Query('full'),
                    tile_size: int = Query(512, ge=128, le=2048), overlap: float = Query(.2, ge=0, le=.5)):
    stored = repo.get_frame(identifier)
    if stored is None: raise HTTPException(404, 'Frame not found')
    if not stored['quality']['passed']: raise HTTPException(422, 'Quality gate failed; recapture before inference')
    if stored['quality'].get('profile', 'camera') not in ('camera', 'neu_proxy'):
        raise HTTPException(422, 'Quality calibration profile does not match the NEU model')
    frame, _ = decode(stored['content'])
    try: result = proxy_detector.detect(frame) if inference_mode == 'full' else proxy_detector.detect(frame, inference_mode, tile_size, overlap)
    except ModelUnavailable as error: raise HTTPException(503, str(error))
    except ValueError as error: raise HTTPException(422, str(error))
    payload = result | {'id': str(uuid4()), 'created_at': utc_now(), 'frame_id': identifier,
                        'image_sha256': stored['sha256'], 'quality': stored['quality']}
    repo.save_model_run(payload)
    return payload

@app.get('/api/proxy/runs/{identifier}')
def model_run(identifier: str):
    record = repo.get_model_run(identifier)
    if record is None: raise HTTPException(404, 'Model run not found')
    return record

@app.post('/api/proxy/frames/{identifier}/anomaly')
def proxy_anomaly(identifier: str, patchcore_model: str = Query('default')):
    detector = patchcore_for(patchcore_model)
    stored = repo.get_frame(identifier)
    if stored is None: raise HTTPException(404, 'Frame not found')
    if not stored['quality']['passed']: raise HTTPException(422, 'Quality gate failed; recapture before inference')
    if stored['quality'].get('profile', 'camera') not in ('camera', 'casting_proxy'):
        raise HTTPException(422, 'Quality calibration profile does not match the casting model')
    frame, _ = decode(stored['content'])
    try: result = detector.detect(frame)
    except PatchCoreUnavailable as error: raise HTTPException(503, str(error))
    except ValueError as error: raise HTTPException(422, str(error))
    payload = result | {'id': str(uuid4()), 'created_at': utc_now(), 'frame_id': identifier,
                        'image_sha256': stored['sha256'], 'quality': stored['quality'],
                        'domain_note': 'MPDD metal-component proxy; use the matching component model. Not brake-disc validated.' if patchcore_model.startswith('mpdd_') else 'Casting-surface proxy. Not validated for brake discs or NEU steel surfaces; no cross-model fusion.',
                        'patchcore_model': patchcore_model}
    repo.save_model_run(payload)
    return payload

DEMO_SETS = {'defect': ('2_casting_anomaly_PatchCore', 'defect_'), 'normal': ('2_casting_anomaly_PatchCore', 'normal_'),
             'verification': ('3_verification_batch_20_normal_castings', ''), 'steel': ('1_steel_defects_YOLO', '')}

def demo_files(name: str) -> list[Path]:
    folder, prefix = DEMO_SETS[name]
    directory = Path(__file__).resolve().parents[2] / 'demo_images' / folder
    return sorted(p for p in directory.glob(prefix + '*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')) if directory.is_dir() else []

@app.get('/api/demo/catalog')
def demo_catalog():
    return {name: [p.name for p in demo_files(name)] for name in DEMO_SETS}

@app.get('/api/demo/file/{set_name}/{file_name}')
def demo_file(set_name: str, file_name: str):
    if set_name not in DEMO_SETS: raise HTTPException(404, 'Unknown demo set')
    match = next((p for p in demo_files(set_name) if p.name == file_name), None)
    if match is None: raise HTTPException(404, 'Demo image not found')
    return Response(match.read_bytes(), media_type='image/png' if match.suffix.lower() == '.png' else 'image/jpeg')

@app.get('/api/patchcore/models')
def patchcore_models():
    mpdd = []
    for category in MPDD_CATEGORIES:
        report_path = MPDD_ROOT / category / 'evaluation.json'
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            mpdd.append({'name':'mpdd_'+category, 'category':category, 'scope':'MPDD component proxy',
                         'artifact_sha256':report['artifact_sha256'], 'test_metrics':report['test'],
                         'brake_disc_validated':False})
    return {'default': {'name': 'default', 'scope': 'Casting impeller dataset (ok_front / def_front)', 'threshold': patchcore_detector.status()['threshold']},
            'custom': patchcore_custom.list_models(), 'mpdd':mpdd}

@app.post('/api/patchcore/custom/{name}/images', status_code=201)
async def patchcore_training_image(name: str, request: Request):
    content, frame, media, quality = await uploaded_frame(request, 'casting_proxy')
    if not quality['passed']:
        raise HTTPException(422, 'Training image rejected by the quality gate: ' + ', '.join(quality['rejection_reasons']) + '. A blurry or blank "normal" would corrupt the model.')
    try: return await run_in_threadpool(patchcore_custom.add_image, name, content, media)
    except ValueError as error: raise HTTPException(422, str(error))
    except FileExistsError as error: raise HTTPException(409, str(error))

@app.post('/api/patchcore/custom/{name}/train', status_code=201)
def patchcore_train(name: str):
    device = patchcore_detector.device
    try: report = patchcore_custom.train_custom(name, device=device)
    except ValueError as error: raise HTTPException(422, str(error))
    except FileExistsError as error: raise HTTPException(409, str(error))
    return {k: v for k, v in report.items() if k != 'metadata'}

@app.get('/api/models')
def model_status():
    analytics = analytics_status()
    assistant_status = assistant.status()
    return {'proxy_detection': proxy_detector.status(), 'rotor_segmentation': {'configured': False},
            'quality_profiles': profile_status(),
            'analytics_pipeline': {'mode': analytics_versions()['analytics'], 'production_validated': False},
            'patchcore': patchcore_detector.status(), 'corrosion': rust_detector.status(), 'xgboost_rca': analytics['xgboost_rca'],
            'plsr': analytics['plsr'], 'pcr': {**analytics['pcr'], 'role': 'benchmark'},
            'llm': {'enabled': assistant_status['llm_configured'], 'read_only': True},
            'voice_alerts': {'enabled': assistant_status['voice_configured'], 'automatic_levels': ['high', 'critical']},
            'supermemory': supermemory.status()}

@app.get('/api/health')
def health():
    from sqlalchemy import text
    try:
        with repo.engine.connect() as connection: connection.execute(text('SELECT 1'))
    except Exception: raise HTTPException(503, 'Database unavailable')
    return {'status': 'ok', 'mode': 'synthetic_replay', 'database': repo.engine.dialect.name,
            'trained_models': proxy_detector.model is not None or analytics_versions()['analytics'] == 'trained-synthetic-v1',
            'auth_enabled': bool(os.getenv('LINEGUARD_API_TOKEN'))}

@app.get('/api/scenarios')
def scenarios():
    return ['thermal_drift', 'cosmetic_scratch', 'normal', 'unknown_anomaly', 'blurred']

@app.post('/api/replay', status_code=201, response_model=InspectionResponse, response_model_exclude_none=True)
def replay(request: ReplayRequest):
    result = create_replay(request)
    job = None
    if request.analytics_mode == 'deferred' and result['quality']['passed']:
        job = {'id': str(uuid4()), 'inspection_id': result['id'], 'status': 'queued', 'created_at': utc_now(), 'error': None}
        result['analytics_job_id'] = job['id']
        result['audit'].append({'at': job['created_at'], 'event': 'analytics_queued', 'job_id': job['id']})
    return repo.save(result, job)

@app.post('/api/analytics/what-if')
def analytics_what_if(request: WhatIfRequest):
    """Same trained analytics path as a real inspection, driven by operator-chosen process readings. Nothing is stored."""
    from analytics.plsr_forecast import load_artifact
    from analytics.monte_carlo import histogram
    telemetry = request.model_dump()
    seed = 2026
    result = analyze_process(telemetry, seed)
    residuals = load_artifact()['residuals']
    result['histogram'] = histogram(result['forecast']['predicted_defect_fraction'], residuals, seed,
                                    interval_scale=result['uncertainty'].get('residual_interval_scale', 1.0))
    result['telemetry'] = telemetry
    result['training_feature_ranges'] = json.loads((Path(__file__).resolve().parents[2] / 'models/xgboost/metadata.json').read_text())['training_feature_ranges']
    return result

@app.get('/api/jobs/{identifier}', response_model=JobResponse)
def job_status(identifier: str):
    job = repo.get_job(identifier)
    if job is None: raise HTTPException(404, 'Analytics job not found')
    return job

@app.post('/api/jobs/{identifier}/retry', response_model=JobResponse)
def retry_job(identifier: str):
    job_status(identifier)
    repo.retry_job(identifier)
    return repo.get_job(identifier)

@app.post('/api/inspections/upload', status_code=201, response_model=InspectionResponse, response_model_exclude_none=True)
async def upload_inspection(request: Request, model: Literal['neu', 'casting'] = Query('casting'),
                            input_source: Literal['upload', 'camera'] = Query('upload'),
                            process_context: Literal['history', 'nominal', 'thermal_drift'] = Query('history'),
                            patchcore_model: str = Query('default'),
                            inference_mode: Literal['full', 'sliced'] = Query('full'),
                            tile_size: int = Query(512, ge=128, le=2048), overlap: float = Query(.2, ge=0, le=.5),
                            part_diameter_mm: float | None = Query(None, gt=0, le=5000),
                            mm_per_px: float | None = Query(None, gt=0, le=100),
                            lot_id: str = Query('LOT-UPLOAD', min_length=1, max_length=40, pattern=r'^[A-Za-z0-9._-]+$'),
                            machine_id: str = Query('M-UPLOAD', min_length=1, max_length=40, pattern=r'^[A-Za-z0-9._-]+$')):
    if model != 'neu' and inference_mode == 'sliced':
        raise HTTPException(422, 'SAHI sliced inference is available only for the NEU YOLO model')
    content, frame, media, quality = await uploaded_frame(request, 'camera' if input_source == 'camera' else PROFILES[model])
    stored = {'id': str(uuid4()), 'created_at': utc_now(), 'sha256': hashlib.sha256(content).hexdigest(),
              'media_type': media, 'content': content, 'quality': quality}
    await run_in_threadpool(repo.save_frame, stored)
    run = run_id = None
    if quality['passed']:
        try:
            if model == 'neu' and inference_mode == 'sliced':
                run = await run_in_threadpool(proxy_detector.detect, frame, inference_mode, tile_size, overlap)
            else:
                run = await run_in_threadpool(proxy_detector.detect if model == 'neu' else patchcore_for(patchcore_model).detect, frame)
        except (ModelUnavailable, PatchCoreUnavailable) as error: raise HTTPException(503, str(error))
        except ValueError as error: raise HTTPException(422, str(error))
        run_id = str(uuid4())
        await run_in_threadpool(repo.save_model_run, run | {'id': run_id, 'created_at': utc_now(), 'frame_id': stored['id'],
                                'image_sha256': stored['sha256'], 'quality': quality})
    # Known-defect detection runs alongside the anomaly model; the pipeline merges the two spatially.
    # Every model scans every quality-passed image so all three views can be shown; only the chosen detector makes findings.
    patchcore_view = None
    if quality['passed'] and model == 'neu':
        try:
            patchcore_view = await run_in_threadpool(patchcore_for(patchcore_model).detect, frame)
        except (PatchCoreUnavailable, ValueError, HTTPException):
            patchcore_view = None
    yolo_run = None
    if quality['passed'] and model == 'casting':
        try:
            yolo_run = await run_in_threadpool(proxy_detector.detect, frame)
        except (ModelUnavailable, ValueError):
            yolo_run = None
    # Corrosion classifier runs on every quality-passed image, whichever defect model was chosen.
    rust_run = None
    if quality['passed']:
        try:
            rust_run = await run_in_threadpool(rust_detector.detect, frame, patchcore_model if model == 'casting' else None)
        except RustUnavailable:
            rust_run = None
    history_context = history_note = None
    if process_context == 'history':
        aggregates = await run_in_threadpool(repo.lot_aggregates, machine_id, 60)
        history_context = build_process_context(aggregates, machine_id)
        if history_context is None:
            history_note = f'No imported sensor history for machine {machine_id}; the nominal simulated preset was used instead.'
        elif lot_id == 'LOT-UPLOAD':
            lot_id = history_context['lot_id']        # trace the part to the machine's current lot
    result = await run_in_threadpool(create_upload_inspection, stored, frame.shape, model, process_context, lot_id, machine_id, run, run_id,
                                     patchcore_model, frame, part_diameter_mm, mm_per_px, history_context, history_note, yolo_run, rust_run, patchcore_view)
    result['context']['input_source'] = input_source
    result['context']['part_identity'] = identify_reference(stored['sha256'], input_source)
    if result['context']['part_identity']['part_type'] == 'brake_disc':
        result['audit'].append({'at': utc_now(), 'event': 'reference_part_identity_matched',
                                'actor': 'reference_registry', **result['context']['part_identity']})
    result['audit'][0]['actor'] = 'camera_scan' if input_source == 'camera' else 'upload'
    try: return await run_in_threadpool(repo.save, result)
    except ValueError as error: raise HTTPException(409, str(error))

@app.get('/api/inspections', response_model=list[InspectionResponse], response_model_exclude_none=True)
def inspections(limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0),
                lot_id: str | None = None, machine_id: str | None = None):
    return repo.list(limit, offset, lot_id, machine_id)

@app.get('/api/inspections/{identifier}', response_model=InspectionResponse, response_model_exclude_none=True)
def inspection(identifier: str):
    result = repo.get(identifier)
    if result is None: raise HTTPException(404, 'Inspection not found')
    return result

@app.get('/api/inspections/{identifier}/image')
def image(identifier: str):
    record = inspection(identifier)
    if record['source'] == 'uploaded_image':
        frame = repo.get_frame(record['context']['frame_id'])
        if frame is None: raise HTTPException(404, 'Source image not found')
        return Response(frame['content'], media_type=frame['media_type'], headers={'ETag': '"'+frame['sha256']+'"', 'X-Content-Type-Options': 'nosniff'})
    return Response(render_svg(record), media_type='image/svg+xml')

@app.get('/api/inspections/{identifier}/audit')
def audit(identifier: str):
    inspection(identifier)
    return {'events': repo.audit(identifier), 'integrity': repo.verify_audit(identifier)}

@app.exception_handler(historical_sensors.SensorCSVError)
async def sensor_csv_error(request, error):
    return JSONResponse({'detail': str(error)}, status_code=422)

@app.post('/api/history/sensor-datasets', status_code=201)
async def import_sensor_dataset(request: Request,
        filename: str = Query('sensor-history.csv', min_length=1, max_length=240),
        source_label: str = Query('source not verified', max_length=180)):
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > historical_sensors.MAX_BYTES:
            raise HTTPException(413, 'CSV exceeds the 5 MiB upload limit.')
        content.extend(chunk)
    if request.headers.get('content-type', '').split(';', 1)[0].lower() not in ('text/csv', 'application/csv', 'application/octet-stream'):
        raise HTTPException(415, 'Upload a UTF-8 CSV file as text/csv.')
    dataset, readings = await run_in_threadpool(historical_sensors.parse_csv, bytes(content), filename, source_label)
    try:
        result = await run_in_threadpool(repo.import_sensor_dataset, dataset, readings)
    except Exception:
        logger.exception('Historical sensor import failed')
        raise HTTPException(503, 'Historical sensor data could not be stored. Check PostgreSQL connectivity and database logs.') from None
    return result | {'machine_count': len({r['machine_id'] for r in readings}),
        'sensor_count': len({r['sensor_name'] for r in readings}), 'provenance': 'User-supplied source label; not independently verified.'}

_risk_cache: dict = {}

@app.get('/api/history/risk-ranking')
def risk_ranking(refresh: bool = False):
    """Ranking takes ~30 s against the remote database (every machine and batch is re-scored), so reuse it for 2 minutes."""
    import time
    from analytics.fleet_risk import rank_machines
    cached = _risk_cache.get('result')
    if cached and not refresh and time.time() - cached[0] < 120:
        return cached[1] | {'cached_seconds_ago': round(time.time() - cached[0])}
    result = rank_machines(repo) | {'generated_at': utc_now()}
    _risk_cache['result'] = (time.time(), result)
    return result

@app.get('/api/history/lot-context')
def lot_context(machine_id: str = Query(..., min_length=1, max_length=80)):
    context = build_process_context(repo.lot_aggregates(machine_id, 60), machine_id)
    if context is None:
        raise HTTPException(404, f'No imported lot history for machine {machine_id}.')
    return context

@app.get('/api/history/sensors/catalog')
def sensor_catalog():
    return repo.sensor_catalog()

@app.get('/api/history/sensors/series')
def sensor_series(machine_id: str = Query(..., min_length=1, max_length=80),
                  sensor: str = Query(..., min_length=1, max_length=80),
                  days: int = Query(30, ge=1, le=90)):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    baseline_start = start - timedelta(days=days)
    current = repo.sensor_history(machine_id, sensor, start, end)
    baseline = repo.sensor_history(machine_id, sensor, baseline_start, start, limit=1)
    delta = None if baseline['mean'] in (None, 0) or current['mean'] is None else (current['mean'] - baseline['mean']) / abs(baseline['mean'])
    return {'machine_id': machine_id, 'sensor': sensor, 'days': days,
        'current_window': {'start': start.isoformat(), 'end': end.isoformat(), **current},
        'previous_window': {'start': baseline_start.isoformat(), 'end': start.isoformat(), **{k:v for k,v in baseline.items() if k != 'points'}},
        'mean_change_fraction': delta, 'source': 'Imported historical sensor observations; source labels are user supplied.'}

@app.get('/api/inspections/{identifier}/briefing')
def briefing(identifier: str): return evidence_packet(inspection(identifier))

async def record_memory_write(identifier: str, kind: str, outcome: dict):
    if outcome.get('reason') == 'not_configured':
        return await run_in_threadpool(repo.get, identifier)     # memory disabled: nothing to record
    def transform(result):
        result['audit'].append({'at': utc_now(), 'event': 'memory_write', 'kind': kind, 'stored': bool(outcome.get('stored')),
                                'reason': outcome.get('reason'), 'container': supermemory.CONTAINER})
        return result
    return await run_in_threadpool(repo.update, identifier, transform)

@app.post('/api/inspections/{identifier}/decision', response_model=InspectionResponse, response_model_exclude_none=True)
async def decision(identifier: str, request: DecisionRequest):
    def transform(result):
        if result['action']['status'] != 'pending':
            raise HTTPException(409, 'Action must be pending to record a decision')
        now = utc_now()
        result['action'].update(status={'approve': 'approved', 'reject': 'rejected', 'escalate': 'escalated'}[request.decision], engineer=request.engineer.strip(), note=request.note, decided_at=now)
        result['audit'].append({'at': now, 'event': 'engineer_decision', **request.model_dump()})
        return result
    try:
        result = await run_in_threadpool(repo.update, identifier, transform)
        if request.decision == 'approve':
            result = await record_memory_write(identifier, 'engineer_approved_action', await supermemory.remember_approved(result))
        return result
    except KeyError: raise HTTPException(404, 'Inspection not found')

@app.post('/api/inspections/{identifier}/verification', response_model=InspectionResponse, response_model_exclude_none=True)
async def verification(identifier: str, request: VerificationRequest):
    def transform(result):
        if result['action']['status'] != 'approved': raise HTTPException(409, 'Engineer approval required')
        if len(set(request.inspection_ids)) != len(request.inspection_ids): raise HTTPException(422, 'Duplicate inspection evidence')
        records = [inspection(i) for i in request.inspection_ids]
        from datetime import datetime
        decision_at = datetime.fromisoformat(result['action'].get('decided_at', result['audit'][-1]['at']))
        if any(datetime.fromisoformat(r['created_at']) <= decision_at or r['machine_id'] != result['machine_id']
               or r['source'] != result['source'] or r['scenario'] == 'blurred' or not r['quality']['passed']
               or (result['source'] == 'uploaded_image' and r['scenario'] != result['scenario']) for r in records):
            raise HTTPException(422, 'Evidence must be later, quality-passed inspections from the same machine and source')
        if len({r['part_id'] for r in records}) != len(records): raise HTTPException(422, 'Verification requires distinct physical part IDs')
        defective = sum(bool(r['defects']) for r in records)
        result['verification'] = {'status': 'demo_criteria_met' if defective == 0 else 'demo_criteria_not_met',
            'parts': len(records), 'defective': defective, 'inspection_ids': request.inspection_ids,
            'observed_sample_defect_rate': defective / len(records),
            'one_sided_95_upper_defect_rate_if_zero': 1 - .05 ** (1 / len(records)) if defective == 0 else None,
            'source': result['source'], 'production_fix_verified': False,
            'note': 'Zero defects in this sample does not establish a production fix.'}
        result['action']['status'] = 'verification_recorded'
        result['audit'].append({'at': utc_now(), 'event': 'verification_recorded', 'evidence': request.inspection_ids})
        return result
    try: result = await run_in_threadpool(repo.update, identifier, transform)
    except KeyError: raise HTTPException(404, 'Inspection not found')
    return await record_memory_write(identifier, 'verification_outcome', await supermemory.remember_verification(result))

@app.get('/')
def index():
    return {'project': 'LineGuard', 'api_docs': '/docs', 'status': 'backend scaffold', 'dashboard': 'http://localhost:3000'}
