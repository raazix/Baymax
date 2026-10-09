"""Declared part identity for exact curated image matches; never model inference."""
import json
from functools import lru_cache
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / 'models/reference_parts/brake_disc.json'


@lru_cache(maxsize=4)
def _read_registry(path: Path, mtime: int):
    table = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(table, dict) or table.get('format_version') != 1 or table.get('part_type') != 'brake_disc':
        return frozenset()
    images = table.get('image_sha256', [])
    if not isinstance(images, list) or any(not isinstance(item, str) or len(item) != 64 for item in images):
        return frozenset()
    return frozenset(images)


def identify_reference(image_sha256: str, input_source: str, registry: Path = REGISTRY) -> dict:
    unknown = {'part_type': 'unclassified', 'source': 'not_identified', 'model_prediction': False,
               'note': 'Part type has not been classified. Heatmaps describe model anomaly evidence.'}
    if input_source != 'upload':
        return unknown
    try:
        known = _read_registry(registry, registry.stat().st_mtime_ns)
    except (OSError, ValueError, TypeError):
        return unknown
    if image_sha256 not in known:
        return unknown
    return {'part_type': 'brake_disc', 'source': 'curated_reference_image', 'model_prediction': False,
            'reference_set': 'dents', 'matched_sha256': image_sha256,
            'note': 'Exact image match to the operator-declared brake-disc reference folder; not a model prediction.'}
