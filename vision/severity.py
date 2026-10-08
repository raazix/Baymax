from core.config import RULE_VERSION

def rate(defect: dict) -> dict:
    label, length, zone = defect['label'], defect['length_mm'], defect['zone']
    if label == 'surface_crack':
        level, reason = ('critical', 'Crack in vane root or length at least 3 mm') if zone == 'vane_root' or length >= 3 else ('high', 'Any detected crack requires review')
    elif label in ('deformation', 'burr'):
        level, reason = 'high', 'Geometry or edge defect requires engineering review'
    elif label == 'anomaly_unclassified':
        level, reason = 'review_required', 'Unknown anomaly cannot be assigned a safe severity'
    elif length >= 2:
        level, reason = 'medium', 'Non-crack defect at least 2 mm long'
    else:
        level, reason = 'low', 'Small non-crack surface defect'
    return {'level': level, 'reason': reason, 'rule_version': RULE_VERSION, 'validated_for_production': False}


# Deterministic demo rules for proxy-model detections. Image pixels only: no rotor zone or physical size is known,
# so these never reach 'critical' and ML confidence is deliberately not an input.
PROXY_BASE = {
    'crazing': ('high', 'Crack-like crazing pattern requires engineering review'),
    'inclusion': ('medium', 'Embedded foreign material requires inspection'),
    'patches': ('low', 'Surface patch defect; confirm material impact on review'),
    'pitted_surface': ('medium', 'Surface pitting requires inspection'),
    'rolled-in_scale': ('low', 'Rolled-in scale requires surface review'),
    'scratches': ('low', 'Surface scratch; depth is not available from this detector'),
    'anomaly_unclassified': ('review_required', 'Unknown anomaly cannot be assigned a safe severity'),
}


def rate_proxy(label: str, *, area_fraction: float | None = None,
               max_side_fraction: float | None = None) -> dict:
    """Conservative proxy triage using class and normalized detector box size.

    The normalized dimensions are the model's bounding box, not a defect mask or a
    physical measurement. This improves triage ordering while retaining explicit
    review limits; it is not a brake-disc safety rating.
    """
    level, reason = PROXY_BASE.get(label, ('review_required', 'Unrecognised defect class cannot be assigned a safe severity'))
    basis = ['defect class']
    if area_fraction is not None and max_side_fraction is not None:
        basis.append('detector bounding-box fraction of image')
        if area_fraction >= .05 or max_side_fraction >= .35:
            level, reason = 'high', 'Large image-relative defect region requires engineering review'
        elif area_fraction >= .01 or max_side_fraction >= .15:
            if level not in ('high', 'review_required'):
                level = 'medium'
            reason = 'Moderate image-relative defect region requires inspection'
    return {'level': level, 'reason': reason + '. ' + '; '.join(basis) + '; image-relative triage only, not a calibrated safety limit.',
            'basis': basis, 'area_fraction': area_fraction, 'max_side_fraction': max_side_fraction,
            'rule_version': RULE_VERSION + '-proxy-v2', 'validated_for_production': False}
