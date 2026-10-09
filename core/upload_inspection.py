"""Build a full inspection record from an uploaded image and a real proxy-model run.

Image-derived: quality gate, defect model output, pixel locations, deterministic severity.
Not image-derived: process telemetry (operator-selected simulated preset) and any millimetre/rotor geometry (omitted).
"""
import hashlib
from uuid import uuid4
from core.clock import utc_now
from vision.severity import rate_proxy, classify_severity
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


def _neu_defects(run: dict, shape, geometry, scale) -> list[dict]:
    defects = []
    height, width = shape[:2]
    for d in run['detections']:
        x1, y1, x2, y2 = d['bbox_xyxy_px']
        w, h = x2 - x1, y2 - y1
        defect = _measure({'label': d['label'], 'confidence': d['confidence'], 'calibrated_confidence': d.get('calibrated_confidence'), 'bbox_xyxy_px': d['bbox_xyxy_px'],
                                 'centroid_px': [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                                 'length_px': round(max(w, h), 1), 'width_px': round(min(w, h), 1), 'area_px': round(w * h, 1),
                                 }, geometry, scale)
        defect['severity'] = rate_proxy(d['label'], area_fraction=min(1.0, max(0.0, w * h / (width * height))),
                                        max_side_fraction=min(1.0, max(w, h) / max(width, height)))
        defects.append(defect)
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


MERGE_MIN_INSIDE = .5      # at least half of the YOLO box must lie inside the anomaly region
MERGE_MIN_CONFIDENCE = .35 # detector's own cut-off for accepting a class name; it never sets severity
MERGE_MAX_REGION = .35     # a flagged region covering more of the image than this is not localised enough to name


def _inside_fraction(box, region):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = region
    inter = max(0, min(x2, rx2) - max(x1, rx1)) * max(0, min(y2, ry2) - max(y1, ry1))
    return inter / max((x2 - x1) * (y2 - y1), 1e-6)


def _merge_detections(defects: list[dict], yolo_run: dict | None, shape=None) -> dict | None:
    """Spatial merge: a YOLO box mostly inside a PatchCore anomaly region names that anomaly's likely class.

    YOLO boxes outside every anomaly region are kept as unconfirmed detections and are not defects: the NEU-trained
    detector also fires on normal castings, so on this domain it may only label what PatchCore already flagged."""
    if not yolo_run:
        return None
    image_area = shape[0] * shape[1] if shape is not None else float('inf')
    detections = []
    for d in yolo_run.get('detections', []):
        detections.append({'label': d['label'], 'confidence': round(d['confidence'], 3), 'bbox_xyxy_px': [round(v, 1) for v in d['bbox_xyxy_px']],
                           'merged_into': None})
    for index, defect in enumerate(defects):
        region = defect.get('region_box_px')
        if not region:
            continue
        x1, y1, x2, y2 = region
        if (x2 - x1) * (y2 - y1) > MERGE_MAX_REGION * image_area:
            defect['class_hint_skipped'] = 'flagged region covers most of the image, so no detection can be attributed to it'
            continue
        matches = [(d, _inside_fraction(d['bbox_xyxy_px'], region)) for d in detections if d['confidence'] >= MERGE_MIN_CONFIDENCE]
        matches = [(d, f) for d, f in matches if f >= MERGE_MIN_INSIDE]
        if matches:
            best, inside = max(matches, key=lambda item: (item[1], item[0]['confidence']))
            best['merged_into'] = index
            defect['class_hint'] = {'label': best['label'], 'confidence': best['confidence'], 'bbox_xyxy_px': best['bbox_xyxy_px'],
                                    'inside_region': round(inside, 2), 'source': 'YOLO11n NEU steel-surface proxy',
                                    'note': 'Class suggested by a detector trained on steel surfaces; confirm visually.'}
    return {'model_sha256': yolo_run.get('model_sha256'), 'detections': detections,
            'merged': sum(d['merged_into'] is not None for d in detections),
            'unconfirmed': sum(d['merged_into'] is None for d in detections),
            'rule': f'box at least {MERGE_MIN_INSIDE:.0%} inside a localised flagged region (<= {MERGE_MAX_REGION:.0%} of the image) and detector confidence >= {MERGE_MIN_CONFIDENCE}',
            'note': 'Unconfirmed detections lie outside every PatchCore-flagged region and are not counted as defects.'}


def _corrosion_defects(rust_run: dict | None, shape, geometry, scale) -> list[dict]:
    """Corrosion finding from the corrosion classifier; its activation map gives a coarse region for measurement."""
    if not rust_run or not rust_run.get('corrosion'):
        return []
    height, width = shape[:2]
    region = anomaly_region(rust_run['activation_map'], .5, width, height) or {
        'centroid_px': [width / 2, height / 2], 'region_box_px': [0, 0, width, height], 'length_px': max(width, height),
        'width_px': min(width, height), 'area_px': width * height, 'cells': 0}
    return [_measure({'label': 'severe_corrosion' if rust_run['severe'] else 'corrosion', 'detector': 'corrosion classifier (ResNet18)',
                      'corrosion_probability': rust_run['corrosion_probability'], 'severe_probability': rust_run['severe_probability'],
                      **region, 'region_basis': 'activation map above half of its peak; coarse attribution, not a mask'}, geometry, scale)]


def _apply_severity(defects: list[dict], geometry, shape) -> None:
    """Severity rules v3 over class (or merged class hint), measured size, location and number of findings."""
    height, width = shape[:2]
    reference = 2 * max(geometry['semi_axes_px']) if geometry else max(width, height)
    for defect in defects:
        hint = defect.get('class_hint')
        label = hint['label'] if hint else defect['label']
        defect['severity'] = classify_severity(
            label, length_mm=defect.get('length_mm'),
            length_fraction=min(1.0, defect['length_px'] / reference) if defect.get('length_px') else None,
            zone=defect.get('zone'), finding_count=len(defects),
            class_source='YOLO class hint merged into anomaly region' if hint else 'model class')


def _anomaly_evidence(run: dict) -> dict:
    """Keep PatchCore's full output whether or not it crosses the threshold, so clean results stay inspectable."""
    return {'score': run['anomaly_score'], 'threshold': run['threshold'], 'flagged': run['anomaly_score'] > run['threshold'],
            'grid': [[round(v, 3) for v in row] for row in run['anomaly_grid']], 'tiling': run.get('tiling'),
            'note': 'Patch distance to the nearest normal training patch; not a probability or segmentation mask.'}


def create_upload_inspection(stored: dict, shape, model: str, process_context: str, lot_id: str, machine_id: str,
                             run: dict | None, run_id: str | None, patchcore_model: str = 'default', frame=None,
                             part_diameter_mm: float | None = None, mm_per_px: float | None = None,
                             history_context: dict | None = None, history_note: str | None = None, yolo_run: dict | None = None,
                             rust_run: dict | None = None, patchcore_view: dict | None = None):
    now = utc_now()
    identifier = str(uuid4())
    sha = stored['sha256']
    passed = bool(stored['quality']['passed'])
    # Polar part coordinates only make sense for round parts (the casting model); steel images are flat patches.
    geometry = locate_part(frame) if passed and frame is not None and model == 'casting' else None
    scale, scale_source = scale_mm_per_px(geometry, part_diameter_mm, mm_per_px)
    defects = [] if not passed or run is None else (_neu_defects(run, shape, geometry, scale) if model == 'neu' else _casting_defects(run, shape, geometry, scale))
    detection_merge = _merge_detections(defects, yolo_run, shape) if model == 'casting' else None
    if passed:
        defects = defects + _corrosion_defects(rust_run, shape, geometry, scale)
    _apply_severity(defects, geometry, shape)
    # Process context: the machine's latest lot from imported history when available; otherwise a labelled preset.
    if history_context:
        telemetry = dict(history_context['telemetry'])
        process_context = 'history'
        telemetry_source = history_context['telemetry_source']
    else:
        telemetry = dict(CONTEXTS['nominal' if process_context == 'history' else process_context])
        telemetry_source = history_note or 'Simulated preset chosen by the operator; NOT measured from the image.'
    vision_sha = '' if run is None else run.get('model_sha256') or run.get('artifact_sha256') or ''
    versions = {'vision': f'uploaded-{model}-proxy:{vision_sha[:12]}', **model_versions()}
    if yolo_run and yolo_run.get('model_sha256'):
        versions['yolo_merge'] = yolo_run['model_sha256']
    if rust_run and rust_run.get('model_sha256'):
        versions['corrosion'] = rust_run['model_sha256']
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
                          'part_shape_hint': 'disc_like_round_outline' if geometry else 'flat_or_unresolved_surface',
                          'routing_note': 'A geometric outline hint only; this does not identify a brake disc. MPDD metal_plate is used as a surface proxy for either input type.',
                          'anomaly': _anomaly_evidence(run) if model == 'casting' and run is not None else (_anomaly_evidence(patchcore_view) if patchcore_view else None),
                          'anomaly_role': 'primary' if model == 'casting' else ('view_only' if patchcore_view else None),
                          'yolo_view': {'role': 'primary' if model == 'neu' else 'hints', 'detections': ([{'label': d['label'], 'confidence': round(d['confidence'], 3), 'calibrated_confidence': d.get('calibrated_confidence'), 'bbox_xyxy_px': [round(v, 1) for v in d['bbox_xyxy_px']]} for d in run['detections']] if model == 'neu' and run else None)},
                          'geometry': geometry,
                          'inference': run.get('inference') if run else None,
                          'detection_merge': detection_merge,
                          'corrosion': {k: rust_run.get(k) for k in ('skipped', 'reason', 'corrosion_probability', 'severe_probability', 'calibrated_corrosion_probability', 'calibrated_severe_probability', 'probability_calibration_note', 'corrosion', 'severe', 'grade', 'thresholds', 'activation_map', 'note')} if rust_run else None,
                          'process_context': process_context,
                          'telemetry_source': telemetry_source,
                          'lot_history': {k: history_context[k] for k in ('lot_window', 'readings', 'prior_lots', 'drift', 'trend', 'baseline_lots', 'synthetic', 'units', 'source_labels')} if history_context else None},
              'audit': [{'at': now, 'event': 'inspection_created', 'actor': 'upload', 'source': 'uploaded_image', 'frame_id': stored['id']}]}
    if passed:
        forecast_input = telemetry | {'history': history_context['forecast_history']} if history_context else telemetry
        result['analytics'] = analyze(forecast_input, seed, versions)
        result['action'] = recommend(defects, result['analytics'])
    else:
        result.update(disposition='recapture', analytics=None,
                      action={'required': False, 'status': 'blocked', 'text': 'Recapture image before inspection.'})
    return result
