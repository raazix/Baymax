import math

def fingerprint(defects: list[dict]) -> dict:
    if not defects:
        return {'count': 0, 'angular_concentration': 0, 'mean_angle_deg': None}
    angles = [math.radians(d['theta_deg']) for d in defects]
    x = sum(map(math.cos, angles)) / len(angles)
    y = sum(map(math.sin, angles)) / len(angles)
    return {'count': len(defects), 'angular_concentration': round(math.hypot(x, y), 3),
            'mean_angle_deg': round(math.degrees(math.atan2(y, x)) % 360, 2),
            'note': 'One part cannot establish a tooling pattern; aggregate across a lot.'}
