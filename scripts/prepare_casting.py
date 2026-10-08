"""Group exact image hashes before splitting casting proxy data."""
from pathlib import Path
import hashlib
import json
import random
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/processed/casting/manifest.json'


def assign_splits(records, seed=42):
    grouped = {}
    for record in records:
        group = grouped.setdefault(record['sha256'], [])
        if group and group[0]['label'] != record['label']:
            raise ValueError('Identical image bytes have conflicting labels')
        group.append(record)
    output = []
    for label in ('normal', 'defect'):
        groups = sorted((items for items in grouped.values() if items[0]['label'] == label), key=lambda items: items[0]['sha256'])
        random.Random(seed).shuffle(groups)
        count = sum(len(items) for items in groups)
        targets = {'train': round(count*.7), 'validation': round(count*.15)} if label == 'normal' else {'validation': count//2}
        targets['test'] = count - sum(targets.values())
        allocated = dict.fromkeys(targets, 0)
        for items in groups:
            split = max(targets, key=lambda name: targets[name]-allocated[name])
            allocated[split] += len(items)
            output.extend({**item, 'split': split} for item in items)
    return sorted(output, key=lambda item: item['path'])


def prepare(seed=42):
    records = []
    with zipfile.ZipFile(ROOT / 'casting_512x512.zip') as archive:
        for name in sorted(archive.namelist()):
            relative = Path(name)
            if relative.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive path')
            label = 'defect' if 'def_front' in relative.parts else 'normal' if 'ok_front' in relative.parts else None
            if label is None:
                raise ValueError(f'Unknown label: {name}')
            content = archive.read(name)
            destination = ROOT / 'data/raw' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.read_bytes() != content:
                destination.write_bytes(content)
            records.append({'path': destination.relative_to(ROOT).as_posix(), 'label': label,
                            'sha256': hashlib.sha256(content).hexdigest(), 'archive_path': name})
    records = assign_splits(records, seed)
    counts = {split: {label: sum(item['split'] == split and item['label'] == label for item in records)
                      for label in ('normal', 'defect')} for split in ('train', 'validation', 'test')}
    manifest = {'seed': seed, 'data_source': 'casting_proxy', 'records': records, 'counts': counts,
                'exact_hash_grouping': True, 'physical_part_ids_available': False,
                'near_duplicate_check_performed': False,
                'limitations': 'Exact-byte duplicates grouped; visually similar images and images of the same physical part may still span splits. Physical part IDs unavailable.'}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'manifest': MANIFEST.relative_to(ROOT).as_posix(), 'counts': counts}, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    prepare()
