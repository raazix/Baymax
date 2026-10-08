import hashlib
import json
import os
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
from database.repository import Repository, ConflictError
from core.clock import utc_now
from vision.image_ingestion import decode, MAX_BYTES
from vision.quality_gate import check_frame, check_frame_open_domain, profile_status, QualityCalibrationUnavailable
from vision.yolo_detector import ProxyDetector, ModelUnavailable
from vision.patchcore_anomaly import PatchCoreDetector, PatchCoreUnavailable
from vision import patchcore_custom
from analytics.briefing import evidence_packet
from analytics import assistant
from pydantic import BaseModel, ConfigDict, Field
from analytics.service import analyze as analyze_process, status as analytics_status, model_versions as analytics_versions, AnalyticsUnavailable

app = FastAPI(title='LineGuard', version='0.1.0', description='Evidence-linked Track 3 hackathon scaffold')

class AssistantQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(default='Explain this inspection and the recommended next action.', min_length=1, max_length=1200)

@app.exception_handler(assistant.AssistantUnavailable)
async def assistant_error(request, error):
    return JSONResponse({'detail': str(error)}, status_code=error.status)

@app.get('/api/assistant/status')
def assistant_status():
    return assistant.status()

@app.post('/api/inspections/{identifier}/assistant')
async def inspection_assistant(identifier: str, question: AssistantQuestion):
    record = await run_in_threadpool(inspection, identifier)
    return await assistant.explain(record, question.question.strip())

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
def proxy_inference(identifier: str):
    stored = repo.get_frame(identifier)
    if stored is None: raise HTTPException(404, 'Frame not found')
    if not stored['quality']['passed']: raise HTTPException(422, 'Quality gate failed; recapture before inference')
    if stored['quality'].get('profile', 'camera') not in ('camera', 'neu_proxy'):
        raise HTTPException(422, 'Quality calibration profile does not match the NEU model')
    frame, _ = decode(stored['content'])
    try: result = proxy_detector.detect(frame)
    except ModelUnavailable as error: raise HTTPException(503, str(error))
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
    return {'proxy_detection': proxy_detector.status(), 'rotor_segmentation': {'configured': False},
            'quality_profiles': profile_status(),
            'analytics_pipeline': {'mode': analytics_versions()['analytics'], 'production_validated': False},
            'patchcore': patchcore_detector.status(), 'xgboost_rca': analytics['xgboost_rca'],
            'plsr': analytics['plsr'], 'pcr': {**analytics['pcr'], 'role': 'benchmark'},
            'llm': {'enabled': False}, 'supermemory': {'enabled': False}}

@app.get('/api/health')
def health():
    from sqlalchemy import text
    try:
        with repo.engine.connect() as connection: connection.execute(text('SELECT 1'))
    except Exception: raise HTTPException(503, 'Database unavailable')
    return {'status': 'ok', 'mode': 'synthetic_replay',
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
    result['histogram'] = histogram(result['forecast']['predicted_defect_fraction'], residuals, seed)
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
                            process_context: Literal['nominal', 'thermal_drift'] = Query('nominal'),
                            patchcore_model: str = Query('default'),
                            part_diameter_mm: float | None = Query(None, gt=0, le=5000),
                            mm_per_px: float | None = Query(None, gt=0, le=100),
                            lot_id: str = Query('LOT-UPLOAD', min_length=1, max_length=40, pattern=r'^[A-Za-z0-9._-]+$'),
                            machine_id: str = Query('M-UPLOAD', min_length=1, max_length=40, pattern=r'^[A-Za-z0-9._-]+$')):
    content, frame, media, quality = await uploaded_frame(request, PROFILES[model])
    stored = {'id': str(uuid4()), 'created_at': utc_now(), 'sha256': hashlib.sha256(content).hexdigest(),
              'media_type': media, 'content': content, 'quality': quality}
    await run_in_threadpool(repo.save_frame, stored)
    run = run_id = None
    if quality['passed']:
        try:
            run = await run_in_threadpool(proxy_detector.detect if model == 'neu' else patchcore_for(patchcore_model).detect, frame)
        except (ModelUnavailable, PatchCoreUnavailable) as error: raise HTTPException(503, str(error))
        except ValueError as error: raise HTTPException(422, str(error))
        run_id = str(uuid4())
        await run_in_threadpool(repo.save_model_run, run | {'id': run_id, 'created_at': utc_now(), 'frame_id': stored['id'],
                                'image_sha256': stored['sha256'], 'quality': quality})
    result = await run_in_threadpool(create_upload_inspection, stored, frame.shape, model, process_context, lot_id, machine_id, run, run_id, patchcore_model, frame, part_diameter_mm, mm_per_px)
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

@app.get('/api/inspections/{identifier}/briefing')
def briefing(identifier: str): return evidence_packet(inspection(identifier))

@app.post('/api/inspections/{identifier}/decision', response_model=InspectionResponse, response_model_exclude_none=True)
def decision(identifier: str, request: DecisionRequest):
    def transform(result):
        if result['action']['status'] != 'pending':
            raise HTTPException(409, 'Action must be pending to record a decision')
        now = utc_now()
        result['action'].update(status={'approve': 'approved', 'reject': 'rejected', 'escalate': 'escalated'}[request.decision], engineer=request.engineer.strip(), note=request.note, decided_at=now)
        result['audit'].append({'at': now, 'event': 'engineer_decision', **request.model_dump()})
        return result
    try: return repo.update(identifier, transform)
    except KeyError: raise HTTPException(404, 'Inspection not found')

@app.post('/api/inspections/{identifier}/verification', response_model=InspectionResponse, response_model_exclude_none=True)
def verification(identifier: str, request: VerificationRequest):
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
    try: return repo.update(identifier, transform)
    except KeyError: raise HTTPException(404, 'Inspection not found')

@app.get('/')
def index():
    return {'project': 'LineGuard', 'api_docs': '/docs', 'status': 'backend scaffold', 'dashboard': 'http://localhost:3000'}
