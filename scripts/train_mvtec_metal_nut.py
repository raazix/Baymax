"""Train a separate PatchCore benchmark from MVTec AD's metal_nut category.

The MVTec archive is non-commercial CC BY-NC-SA 4.0. This script extracts only
metal_nut into ignored local data directories; it never adds the archive/images
or the trained model artifact to Git.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision import patchcore_custom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT / 'archive.zip')
    parser.add_argument('--name', default='mvtec_metal_nut')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    archive = args.archive.resolve()
    if not archive.is_file():
        raise SystemExit(f'MVTec archive not found: {archive}')
    output_root = ROOT / 'data/processed/mvtec_ad'
    training_root = ROOT / 'data/custom_training' / args.name
    category_prefix = 'metal_nut/'

    with zipfile.ZipFile(archive) as bundle:
        members = [item for item in bundle.infolist() if item.filename.startswith(category_prefix) and not item.is_dir()]
        if not members:
            raise SystemExit('No metal_nut category found in archive.zip.')
        for item in members:
            relative = Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise SystemExit(f'Unsafe archive path: {item.filename}')
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(bundle.read(item))  # also checks each extracted member CRC

    normal_images = sorted((output_root / 'metal_nut/train/good').glob('*.png'))
    if len(normal_images) < patchcore_custom.MIN_IMAGES:
        raise SystemExit(f'Expected at least {patchcore_custom.MIN_IMAGES} normal training images; found {len(normal_images)}.')
    training_root.mkdir(parents=True, exist_ok=True)
    for source in normal_images:
        content = source.read_bytes()
        destination = training_root / f'{hashlib.sha256(content).hexdigest()}.png'
        if not destination.exists():
            destination.write_bytes(content)

    report = patchcore_custom.train_custom(args.name, device=args.device)
    print(f"Trained {report['name']}: {report['train_images']} memory-bank normals, "
          f"{report['heldout_images']} held-out normals, threshold {report['threshold']:.4f}, "
          f"artifact SHA-256 {report['artifact_sha256']}")
    print('The official MVTec test split remains untouched for evaluation. This model is not brake-disc validated.')


if __name__ == '__main__':
    main()
