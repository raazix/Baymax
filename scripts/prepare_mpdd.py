"""Validate/extract MPDD; preserve official test and split training normals only."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import zipfile
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/processed/mpdd/manifest.json'

def prepare():
    if MANIFEST.exists(): raise FileExistsError('Preserve existing MPDD manifest')
    raw = ROOT / 'data/raw'
    records, files = [], {}
    with zipfile.ZipFile(ROOT / 'MPDD.zip') as archive:
        for index, info in enumerate(archive.infolist()):
            if info.is_dir(): continue
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in info.filename or relative.parts[0] != 'MPDD':
                raise ValueError('Unsafe or unexpected MPDD archive path')
            target = raw.joinpath(*relative.parts).resolve()
            if not target.is_relative_to((raw / 'MPDD').resolve()): raise ValueError('Archive extraction escaped dataset')
            content = archive.read(info)  # ZIP reader checks each entry CRC.
            if relative.suffix.lower() not in ('.png', '.jpg', '.jpeg'): continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != content: raise ValueError('Existing extracted data differ from archive')
            if not target.exists(): target.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            files[info.filename] = (target, digest)
            if len(relative.parts) != 5: raise ValueError('Unexpected MPDD image layout')
            _, category, source_split, kind, filename = relative.parts
            if source_split == 'ground_truth': continue
            if source_split not in ('train', 'test') or source_split == 'train' and kind != 'good':
                raise ValueError('Memory-bank source must contain official training normals only')
            frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None: raise ValueError('Unreadable dataset image')
            records.append({'path': target.relative_to(ROOT).as_posix(), 'category':category,
                'source_split':source_split, 'split':'test' if source_split == 'test' else None,
                'label':'normal' if kind == 'good' else 'defect', 'defect_type':kind,
                'sha256':digest, 'height':int(frame.shape[0]), 'width':int(frame.shape[1]),
                'mask_path':None, 'mask_sha256':None})
            if len(records) % 200 == 0: print(f'Validated {len(records)} MPDD images', flush=True)
    groups = defaultdict(list)
    for record in records: groups[record['sha256']].append(record)
    for items in groups.values():
        if len({r['source_split'] for r in items}) != 1:
            raise ValueError('Official train/test contain an exact duplicate; isolate it before training')
        if len({(r['category'],r['label']) for r in items}) != 1:
            raise ValueError('Conflicting labels or components for identical image content')
    categories = sorted({r['category'] for r in records})
    for category in categories:
        normal_groups = sorted((g for g in groups.values() if g[0]['category'] == category and g[0]['source_split'] == 'train'),key=lambda g:g[0]['sha256'])
        random.Random(42).shuffle(normal_groups)
        validation_target = max(1, round(sum(map(len,normal_groups)) * .2))
        count = 0
        for group in normal_groups:
            split = 'validation' if count < validation_target else 'train'
            if split == 'validation': count += len(group)
            for r in group: r['split'] = split
        if not any(r['category'] == category and r['split'] == 'train' for r in records): raise ValueError('No training normals')
    for r in records:
        if r['label'] == 'defect':
            source = PurePosixPath(r['path']).name
            mask_name = f"MPDD/{r['category']}/ground_truth/{r['defect_type']}/{Path(source).stem}_mask.png"
            if mask_name not in files: raise ValueError('Defective image is missing its genuine mask')
            mask_path, mask_digest = files[mask_name]
            mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (r['height'],r['width']) or not np.any(mask):
                raise ValueError('Mask dimensions/content do not match defective image')
            r.update(mask_path=mask_path.relative_to(ROOT).as_posix(),mask_sha256=mask_digest)
    counts = {category:{split:{label:sum(r['category']==category and r['split']==split and r['label']==label for r in records)
                              for label in ('normal','defect')} for split in ('train','validation','test')} for category in categories}
    manifest = {'dataset':'MPDD','seed':42,'categories':categories,'counts':counts,
                'records':sorted(records,key=lambda r:r['path']), 'official_test_preserved':True,
                'exact_hash_groups_kept_together':True,'duplicate_groups':sum(len(g)>1 for g in groups.values()),
                'threshold_selection':'Held-out official training normals only; test labels/masks excluded.',
                'limitations':'Metal-component anomaly proxy, not brake discs. Physical part identities and near-duplicate audit unavailable.'}
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'categories':categories,'counts':counts,'image_count':len(records),'masks':sum(r['mask_path'] is not None for r in records)},indent=2),flush=True)

if __name__ == '__main__': prepare()
