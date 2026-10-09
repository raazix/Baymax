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


# ---------------------------------------------------------------------------------------------------------------
# Severity rules v3: deterministic matrix over defect class, measured size, location and multiplicity.
# Model confidence and anomaly score are deliberately NOT inputs. Thresholds are demo engineering assumptions for a
# brake-disc-like round part, documented in docs/SEVERITY.md; they are not validated production limits.
LEVELS = ('low', 'medium', 'high', 'critical')
CLASS_BASE = {
    'crazing': ('high', 'crack-like defect'), 'surface_crack': ('high', 'crack'),
    'inclusion': ('medium', 'embedded inclusion'), 'pitted_surface': ('medium', 'surface pitting'),
    'patches': ('low', 'surface patch'), 'rolled-in_scale': ('low', 'rolled-in scale'),
    'scratches': ('low', 'surface scratch'), 'scratch': ('low', 'surface scratch'),
    'anomaly_unclassified': ('medium', 'unclassified anomaly'),
    'corrosion': ('medium', 'surface corrosion'), 'severe_corrosion': ('high', 'severe corrosion'),
}
CRACK_LIKE = {'crazing', 'surface_crack'}
SIZE_MM = ((15.0, 2), (5.0, 1))             # longest side in mm -> escalation steps
SIZE_FRACTION = ((.35, 2), (.15, 1))         # longest side relative to part diameter (or frame) when no scale
CRITICAL_ZONES = {'rim': 'rim / friction edge', 'bore / hub': 'hub / mounting area'}
SEVERITY_VERSION = 'severity-rules-v3'


def classify_severity(label: str, *, length_mm: float | None = None, length_fraction: float | None = None,
                      zone: str | None = None, finding_count: int = 1, class_source: str = 'model class') -> dict:
    base, description = CLASS_BASE.get(label, ('medium', 'unrecognised defect class'))
    index = LEVELS.index(base)
    factors = [{'factor': 'class', 'value': description, 'effect': f'base {base}', 'source': class_source}]
    steps = 0
    if length_mm is not None:
        steps = next((s for limit, s in SIZE_MM if length_mm >= limit), 0)
        factors.append({'factor': 'size', 'value': f'{length_mm:.1f} mm longest side', 'effect': f'+{steps}' if steps else 'none'})
    elif length_fraction is not None:
        steps = next((s for limit, s in SIZE_FRACTION if length_fraction >= limit), 0)
        factors.append({'factor': 'size', 'value': f'{length_fraction:.0%} of part/frame', 'effect': f'+{steps}' if steps else 'none'})
    index += steps
    if zone in CRITICAL_ZONES and (label in CRACK_LIKE or label == 'anomaly_unclassified'):
        index += 1
        factors.append({'factor': 'location', 'value': CRITICAL_ZONES[zone], 'effect': '+1'})
    elif zone and zone not in ('not_measured',):
        factors.append({'factor': 'location', 'value': zone, 'effect': 'none'})
    if finding_count >= 3:
        index += 1
        factors.append({'factor': 'multiplicity', 'value': f'{finding_count} findings on this part', 'effect': '+1'})
    cap = 'critical'
    if label == 'anomaly_unclassified':
        cap = 'high'          # an unknown anomaly is never automatically critical; an engineer must classify it
    elif label not in CRACK_LIKE:
        cap = 'high'          # only crack-like defects can be critical
    physical = length_mm is not None or zone in CRITICAL_ZONES
    if cap == 'critical' and not physical:
        cap = 'high'          # critical needs a measured size (mm) or a located critical zone, not image pixels alone
    capped = index > LEVELS.index(cap) and cap != 'critical'   # 'critical' is simply the top of the scale
    level = LEVELS[min(index, LEVELS.index(cap))]
    reason = f"{level.capitalize()}: {description}" + ''.join(
        f", {f['value']} ({f['effect']})" for f in factors[1:] if f['effect'] not in ('none',))
    if capped:
        if label == 'anomaly_unclassified':
            reason += f'; capped at {cap} until an engineer classifies the anomaly'
        elif label in CRACK_LIKE:
            reason += f'; capped at {cap} because critical needs a measured size or a located rim/hub zone'
        else:
            reason += f'; capped at {cap} because only crack-like defects can be critical'
    return {'level': level, 'reason': reason + '.', 'factors': factors, 'rule_version': SEVERITY_VERSION,
            'review_required': label == 'anomaly_unclassified' or level in ('high', 'critical'),
            'inputs_excluded': 'model confidence and anomaly score are not severity inputs',
            'validated_for_production': False}
