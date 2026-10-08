import hashlib
from uuid import uuid4
from core.clock import utc_now
from core.schemas import Detection, ReplayRequest
from vision.calibration import mm_per_pixel
from vision.metrology import measure
from vision.severity import rate
from analytics.service import analyze, model_versions
from analytics.spatial_fingerprint import fingerprint
from actions.sop_engine import recommend

def create_replay(request: ReplayRequest):
    labels = {'thermal_drift': 'surface_crack', 'cosmetic_scratch': 'scratch', 'unknown_anomaly': 'anomaly_unclassified'}
    label = labels.get(request.scenario)
    scale = mm_per_pixel(16.1)  # 512px illustration: marker spans 16.1px, not original 161px camera example.
    defects = []
    if label:
        detection = Detection(label=label, confidence=.93 if label == 'surface_crack' else .65,
            centroid_px=(168, 166), area_px=8, length_px=6 if label == 'surface_crack' else 2,
            polygon_px=[(164, 162), (166, 161), (173, 171), (171, 173)])
        defect = measure(detection, scale)
        defect['severity'] = rate(defect)
        defects.append(defect)
    now = utc_now()
    identifier = str(uuid4())
    thermal = request.scenario == 'thermal_drift'
    telemetry = {'temperature_c': 758 if thermal else 705, 'pressure_bar': 113 if thermal else 101,
                 'vibration_mm_s': 4.1 if thermal else 2.1, 'machine_speed_rpm': 1200}
    passed = request.scenario != 'blurred'
    result = {'id': identifier, 'created_at': now, 'source': 'synthetic_replay', 'scenario': request.scenario,
        'seed': request.seed, 'part_id': 'P-' + identifier[:8], 'lot_id': 'B127', 'machine_id': 'M-04',
        'quality': {'passed': passed, 'source': 'scenario fixture; not measured camera quality'},
        'calibration': {'marker_mm': 10, 'marker_px': 16.1, 'mm_per_px': scale, 'source': 'synthetic geometry', 'center_px': [256, 256]},
        'defects': defects, 'telemetry': telemetry, 'spatial_fingerprint': fingerprint(defects),
        'disposition': 'review' if defects else 'pass',
        'image_url': '/api/inspections/' + identifier + '/image', 'model_versions': {'vision': 'replay-fixture-v1', **model_versions()},
        'audit': [{'at': now, 'event': 'inspection_created', 'actor': 'replay', 'source': 'synthetic_replay'}]}
    if passed and request.analytics_mode == 'deferred':
        result.update(analytics=None, action={'required': False, 'status': 'awaiting_analytics', 'text': 'Analytics queued; engineering recommendation pending.'})
    elif passed:
        result['analytics'] = analyze(telemetry, request.seed, result['model_versions'])
        result['action'] = recommend(defects, result['analytics'])
    else:
        result.update(disposition='recapture', analytics=None, action={'required': False, 'status': 'blocked', 'text': 'Recapture image before inspection.'})
    result['image_sha256'] = hashlib.sha256(render_svg(result).encode()).hexdigest()
    return result

def render_svg(result):
    # Synthetic illustrative geometry, not a camera photograph or inference result.
    marks = ''.join('<polygon points="' + ' '.join(f'{x},{y}' for x,y in d['polygon_px']) + '" fill="#c04432" stroke="#c04432" stroke-width="3"/>' for d in result['defects'])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" fill="#eae9e3"/><g fill="none" stroke="#8b918f"><circle cx="256" cy="256" r="218" stroke-width="4"/><circle cx="256" cy="256" r="205" stroke-width="15"/><circle cx="256" cy="256" r="170" stroke-width="55"/><circle cx="256" cy="256" r="140"/><circle cx="256" cy="256" r="72" stroke-width="30"/><circle cx="256" cy="256" r="42" stroke-width="3"/><path d="M256 25V487M25 256H487" stroke-dasharray="4 8" opacity=".3"/></g>{marks}<rect x="450" y="460" width="16.1" height="8" fill="#242c2c"/><text x="385" y="493" font-family="sans-serif" font-size="12">10 mm fixture</text><text x="20" y="25" font-family="sans-serif" font-size="12">SYNTHETIC ROTOR / NOT A PHOTOGRAPH</text></svg>'''
