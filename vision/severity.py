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
PROXY_RULES = {
    'crazing': ('high', 'Crack-like crazing network requires engineering review'),
    'inclusion': ('medium', 'Embedded foreign material requires inspection'),
    'patches': ('medium', 'Surface patch defect requires inspection'),
    'pitted_surface': ('medium', 'Surface pitting requires inspection'),
    'rolled-in_scale': ('low', 'Rolled-in scale is typically cosmetic; confirm on review'),
    'scratches': ('low', 'Surface scratch; confirm depth on review'),
    'anomaly_unclassified': ('review_required', 'Unknown anomaly cannot be assigned a safe severity'),
}


def rate_proxy(label: str) -> dict:
    level, reason = PROXY_RULES.get(label, ('review_required', 'Unrecognised defect class cannot be assigned a safe severity'))
    return {'level': level, 'reason': reason + '. Set by defect class only; size and position are reported but not yet used in the rule.',
            'rule_version': RULE_VERSION + '-proxy', 'validated_for_production': False}
