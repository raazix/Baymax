"""Validate local NEU-DET and prepare a reproducible YOLO detection proxy split.

Exact duplicate images are grouped into a single split. Physical-part identities
are unavailable, so this is an image-group split, not production validation.
"""
import argparse
import collections
import hashlib
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def prepare(archive: Path, output: Path):
    import cv2
    import numpy as np
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new output directory to preserve existing splits.')
    records = []
    filename_mismatches = []
    with zipfile.ZipFile(archive) as zipped:
        for entry in sorted(zipped.namelist()):
            if not entry.lower().endswith('.xml'): continue
            root = ET.fromstring(zipped.read(entry))
            stem = PurePosixPath(entry).stem
            image_name = f'NEU-DET/IMAGES/{stem}.jpg'
            payload = zipped.read(image_name)
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None: raise ValueError(f'Corrupt image: {image_name}')
            height, width = image.shape[:2]
            if (width, height) != (int(root.findtext('size/width')), int(root.findtext('size/height'))):
                raise ValueError(f'Annotation/image size mismatch: {entry}')
            if root.findtext('filename') != stem + '.jpg': filename_mismatches.append(entry)
            labels = []
            for obj in root.findall('object'):
                name = obj.findtext('name')
                box = obj.find('bndbox')
                x1, y1, x2, y2 = [float(box.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f'Invalid box: {entry}')
                # Preserve NEU's supplied continuous box extents; no invented masks.
                labels.append((CLASSES.index(name), (x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height))
            records.append({'stem': stem, 'class': stem.rsplit('_', 1)[0], 'bytes': payload,
                            'sha256': hashlib.sha256(payload).hexdigest(), 'labels': labels})
    grouped = collections.defaultdict(list)
    for record in records: grouped[record['sha256']].append(record)
    buckets = collections.defaultdict(list)
    for digest, group in grouped.items(): buckets[group[0]['class']].append(digest)
    assignments = {}
    rng = random.Random(42)
    for category in sorted(buckets):
        keys = sorted(buckets[category]); rng.shuffle(keys)
        train_end, val_end = int(len(keys)*.7), int(len(keys)*.85)
        for i, key in enumerate(keys): assignments[key] = 'train' if i < train_end else 'val' if i < val_end else 'test'
    output.mkdir(parents=True)
    counts = collections.Counter(); boxes = collections.Counter(); manifest = []
    for record in records:
        split = assignments[record['sha256']]
        image_dir, label_dir = output / 'images' / split, output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / (record['stem']+'.jpg')).write_bytes(record['bytes'])
        lines = [' '.join([str(label[0]), *(f'{value:.8f}' for value in label[1:])]) for label in record['labels']]
        (label_dir / (record['stem']+'.txt')).write_text('\n'.join(lines)+'\n', encoding='utf-8')
        counts[split]+=1
        for label in record['labels']: boxes[CLASSES[label[0]]] += 1
        manifest.append({'image': record['stem']+'.jpg', 'split': split, 'sha256': record['sha256']})
    yaml = f'path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n' + ''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES))
    (output / 'dataset.yaml').write_text(yaml, encoding='utf-8')
    report = {'source': archive.name, 'task': 'steel_surface_detection_proxy', 'seed': 42,
              'images': len(records), 'split_counts': dict(counts), 'boxes_by_class': dict(boxes),
              'xml_filename_fields_corrected': len(filename_mismatches),
              'exact_duplicate_groups': sum(len(group)>1 for group in grouped.values()),
              'split_strategy': '70/15/15 by exact-image-hash groups, stratified by filename class; no physical-part metadata',
              'bounding_box_convention': 'Original XML extents normalised continuously; no masks generated',
              'manifest': manifest}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'manifest'}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path('NEU-DET.zip'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/neu-det'))
    args = parser.parse_args()
    prepare(args.archive, args.output)
