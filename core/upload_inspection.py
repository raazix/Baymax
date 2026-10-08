"""Build a full inspection record from an uploaded image and a real proxy-model run.

Image-derived: quality gate, defect model output, pixel locations, deterministic severity.
Not image-derived: process telemetry (operator-selected simulated preset) and any millimetre/rotor geometry (omitted).
"""
import hashlib
from uuid import uuid4
from core.clock import utc_now
from vision.severity import rate_proxy
from vision.part_geometry import locate_part, polar_position, scale_mm_per_px, anomaly_region
from analytics.spatial_fingerprint import fingerprint
from analytics.service import analyze, model_versions
from actions.sop_engine import recommend

CONTEXTS = {
    'nominal': {'temperature_c': 705, 'pressure_bar': 101, 'vibration_mm_s': 2.1, 'machine_speed_rpm': 1200},
    'thermal_drift': {'temperature_c': 758, 'pressure_bar': 113, 'vibration_mm_s': 4.1, 'machine_speed_rpm': 1200},
}
PROFILES = {'neu': 'neu_proxy', 'casting': 'casting_proxy'}
def part_identifier(sha: str, lot_id: str, machine_id: str) -> str:
    # Same image in the same lot/machine is a repeat inspection; in another lot it is a different part.
    return 'IMG-' + sha[:8] + '-' + hashlib.sha256(f'{lot_id}|{machine_id}'.encode()).hexdigest()[:4]


def _measure(defect: dict, geometry, scale) -> dict:
    """Pixel size always; polar position when the part outline is located; millimetres only with an operator scale."""
    x, y = defect['centroid_px']
    if geometry:
        defect.update(polar_position(geometry, x, y))
    else:
        defect['zone'] = 'not_measured'
    if scale:
        defect['length_mm'] = round(defect['length_px'] * scale, 2)
        defect['width_mm'] = round(defect['width_px'] * scale, 2)
        if geometry:
            defect['r_mm'] = round(defect['r_fraction'] * max(geometry['semi_axes_px']) * scale, 1)
    basis = ['size in image pixels']
    basis.append('position in polar part coordinates from the fitted part outline' if geometry else 'no part outline located, so no radial position')
    basis.append('millimetres from the operator-entered scale' if scale else 'no millimetres without an operator-entered scale')
    defect['measurement_basis'] = '; '.join(basis)
    return defect


def _neu_defects(run: dict, geometry, scale) -> list[dict]:
    defects = []
    for d in run['detections']:
        x1, y1, x2, y2 = d['bbox_xyxy_px']
        w, h = x2 - x1, y2 - y1
        defects.append(_measure({'label': d['label'], 'confidence': d['confidence'], 'bbox_xyxy_px': d['bbox_xyxy_px'],
                                 'centroid_px': [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                                 'length_px': round(max(w, h), 1), 'width_px': round(min(w, h), 1), 'area_px': round(w * h, 1),
                                 'severity': rate_proxy(d['label'])}, geometry, scale))
    return defects


def _casting_defects(run: dict, shape, geometry, scale) -> list[dict]:
    grid, threshold = run['anomaly_grid'], run['threshold']
    if run['anomaly_score'] <= threshold:
        return []
    rows, cols = len(grid), len(grid[0])
    height, width = shape[:2]
    region = anomaly_region(grid, threshold, width, height)
    over = sum(v > threshold for row in grid for v in row)
    return [_measure({'label': 'anomaly_unclassified', 'anomaly_score': run['anomaly_score'], 'threshold': threshold,
                      'cells_over_threshold': over, 'grid_cells': rows * cols, **region,
                      'severity': rate_proxy('anomaly_unclassified')}, geometry, scale)]


def _anomaly_evidence(run: dict) -> dict:
    """Keep PatchCore's full output whether or not it crosses the threshold, so clean results stay inspectable."""
    return {'score': run['anomaly_score'], 'threshold': run['threshold'], 'flagged': run['anomaly_score'] > run['threshold'],
            'grid': [[round(v, 3) for v in row] for row in run['anomaly_grid']], 'tiling': run.get('tiling'),
            'note': 'Patch distance to the nearest normal training patch; not a probability or segmentation mask.'}


def create_upload_inspection(stored: dict, shape, model: str, process_context: str, lot_id: str, machine_id: str,
                             run: dict | None, run_id: str | None, patchcore_model: str = 'default', frame=None,
                             part_diameter_mm: float | None = None, mm_per_px: float | None = None):
    now = utc_now()
    identifier = str(uuid4())
    sha = stored['sha256']
    passed = bool(stored['quality']['passed'])
    # Polar part coordinates only make sense for round parts (the casting model); steel images are flat patches.
    geometry = locate_part(frame) if passed and frame is not None and model == 'casting' else None
    scale, scale_source = scale_mm_per_px(geometry, part_diameter_mm, mm_per_px)
    defects = [] if not passed or run is None else (_neu_defects(run, geometry, scale) if model == 'neu' else _casting_defects(run, shape, geometry, scale))
    telemetry = dict(CONTEXTS[process_context])
    vision_sha = '' if run is None else run.get('model_sha256') or run.get('artifact_sha256') or ''
    versions = {'vision': f'uploaded-{model}-proxy:{vision_sha[:12]}', **model_versions()}
    seed = int(sha[:8], 16) % 2147483647
    result = {'id': identifier, 'created_at': now, 'source': 'uploaded_image', 'scenario': f'upload_{model}', 'seed': seed,
              'part_id': part_identifier(sha, lot_id, machine_id), 'lot_id': lot_id, 'machine_id': machine_id, 'quality': stored['quality'],
              'calibration': {'source': scale_source or 'none', 'mm_per_px': round(scale, 5) if scale else None,
                              'part_outline': 'located' if geometry else 'not located',
                              'note': ('Millimetres use the operator-entered scale; accuracy depends on that value and on camera tilt.' if scale
                                       else 'No scale entered: sizes are in pixels.') + (' Radial position is relative to the fitted part outline.' if geometry
                                       else ' Part outline not located reliably, so no radial position.')},
              'defects': defects, 'telemetry': telemetry,
              'spatial_fingerprint': fingerprint(defects) if defects and all('theta_deg' in d for d in defects) else
                                     {'count': len(defects), 'note': 'Angular fingerprint needs a located part outline.'},
              'disposition': 'review' if defects else 'pass', 'image_url': f'/api/inspections/{identifier}/image', 'image_sha256': sha,
              'model_versions': versions,
              'context': {'model': model, 'frame_id': stored['id'], 'model_run_id': run_id, 'image_size_px': [shape[1], shape[0]], 'patchcore_model': patchcore_model if model == 'casting' else None,
                          'anomaly': _anomaly_evidence(run) if model == 'casting' and run is not None else None,
                          'geometry': geometry,
                          'inference': run.get('inference') if run else None,
                          'process_context': process_context,
                          'telemetry_source': 'Simulated preset chosen by the operator; NOT measured from the image.'},
              'audit': [{'at': now, 'event': 'inspection_created', 'actor': 'upload', 'source': 'uploaded_image', 'frame_id': stored['id']}]}
    if passed:
        result['analytics'] = analyze(telemetry, seed, versions)
        result['action'] = recommend(defects, result['analytics'])
    else:
        result.update(disposition='recapture', analytics=None,
                      action={'required': False, 'status': 'blocked', 'text': 'Recapture image before inspection.'})
    return result
