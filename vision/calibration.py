import math

def mm_per_pixel(marker_pixels: float, marker_mm: float = 10) -> float:
    if not all(math.isfinite(x) and x > 0 for x in (marker_pixels, marker_mm)):
        raise ValueError('Marker dimensions must be positive and finite')
    return marker_mm / marker_pixels
