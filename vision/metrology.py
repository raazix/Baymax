import math
from core.schemas import Detection

def measure(detection: Detection, scale: float, center=(256, 256)) -> dict:
    dx = detection.centroid_px[0] - center[0]
    dy = center[1] - detection.centroid_px[1]
    radius = math.hypot(dx, dy) * scale
    zone = 'mounting_hat' if radius < 45 else 'vane_root' if radius < 90 else 'friction_face'
    return {**detection.model_dump(mode='json'), 'length_mm': round(detection.length_px * scale, 3),
            'equivalent_diameter_mm': round(2 * math.sqrt(detection.area_px / math.pi) * scale, 3),
            'r_mm': round(radius, 2), 'theta_deg': round(math.degrees(math.atan2(dy, dx)) % 360, 2),
            'zone': zone, 'coordinate_convention': '0 degrees right; counterclockwise; image y inverted'}
