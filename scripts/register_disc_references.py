"""Register declared brake-disc reference identities without copying images or defect labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.part_reference import REGISTRY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if not folder.is_dir():
        parser.error('Reference folder does not exist')
    images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in
                    {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp'})
    if not images:
        parser.error('Reference folder contains no supported image files')
    digests = sorted({hashlib.sha256(path.read_bytes()).hexdigest() for path in images})
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps({'format_version': 1, 'part_type': 'brake_disc',
        'source': 'operator_declared_reference_folder', 'reference_set': 'dents',
        'model_prediction': False, 'image_sha256': digests}, indent=2) + '\n', encoding='utf-8')
    print(f'Registered {len(digests)} unique reference images from {len(images)} files.')


if __name__ == '__main__':
    main()
