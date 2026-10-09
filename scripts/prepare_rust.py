"""Prepare the Roboflow "Rust Detection" (CC BY 4.0) export for honest training and evaluation.

The export's splits are not usable as-is: its training images are 2x2 mosaics that stitch tiles from several different
photos (labels are the union of the tiles), the supplied test split also contains mosaics, and every validation/test
source photo also appears in train. This script:

1. detects mosaics by their full-width horizontal and full-height vertical seams and excludes them;
2. groups the remaining single-photo images by source photo (Roboflow filename stem);
3. derives image-level targets: corrosion present, and worst grade present (severe vs not severe);
4. splits by source photo (seeded 60/20/20), so no photo or augmented variant crosses splits.

Output: data/processed/rust/manifest.json (paths, hashes, targets, mosaic scores, split).
"""
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/rust_detection'
OUT = ROOT / 'data/processed/rust'
CORROSION = ('Rust', 'copper corrosion', 'corroded-part', 'corrosion', 'iron rust', 'mild-corrosion',
             'moderate-corrosion', 'rust', 'severe-corrosion')
SEAM_FRACTION = .45   # share of a row/column that must show a sharp step for it to count as a seam


def source(name):
    return re.split(r'_(?:png|jpg|jpeg)_jpg\.rf\.|\.rf\.', name)[0]


def seam_scores(path):
    """Strongest straight horizontal and vertical step across the interior of the image (0-1)."""
    gray = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (320, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
    rows = (np.abs(np.diff(gray, axis=0)) > 22).mean(axis=1)[16:-16]     # step between consecutive rows
    cols = (np.abs(np.diff(gray, axis=1)) > 22).mean(axis=0)[16:-16]
    # a seam is a line far stronger than its neighbours (texture edges are not straight across the whole image)
    def peak(profile):
        i = int(np.argmax(profile))
        neighbours = np.r_[profile[max(0, i - 6):max(0, i - 2)], profile[i + 3:i + 7]]
        return float(profile[i]), float(profile[i] - (neighbours.mean() if neighbours.size else 0))
    return peak(rows), peak(cols)


def is_mosaic(scores):
    (row, row_gap), (col, col_gap) = scores
    both = row >= SEAM_FRACTION and col >= SEAM_FRACTION and row_gap >= .25 and col_gap >= .25
    one_strong = (row >= .38 and row_gap >= .25) or (col >= .38 and col_gap >= .25)
    # Excluding a real photo only costs data; keeping a mosaic would leak other photos into its split, so err strict.
    return both or one_strong


def targets(labels):
    present = any(labels.get(c) for c in CORROSION)
    grade = 'severe' if labels.get('severe-corrosion') else 'moderate' if labels.get('moderate-corrosion') else \
        'mild' if labels.get('mild-corrosion') else ('ungraded' if present else 'none')
    return present, grade


def main(seed=42):
    records = []
    for split in ('train', 'valid', 'test'):
        with open(RAW / split / '_classes.csv', encoding='utf-8') as handle:
            table = list(csv.reader(handle))
        header = [h.strip() for h in table[0]][1:]
        for row in table[1:]:
            if not row:
                continue
            name = row[0].strip()
            labels = dict(zip(header, (int(v) for v in row[1:])))
            path = RAW / split / name
            scores = seam_scores(path)
            present, grade = targets(labels)
            records.append({'path': path.relative_to(ROOT).as_posix(), 'original_split': split, 'source': source(name),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'labels': [k for k, v in labels.items() if v],
                            'corrosion': present, 'grade': grade, 'severe': grade == 'severe',
                            'seam': [round(scores[0][0], 3), round(scores[1][0], 3)], 'mosaic': is_mosaic(scores)})
    clean = [r for r in records if not r['mosaic']]
    groups = defaultdict(list)
    for r in clean:
        groups[r['source']].append(r)
    # exact duplicates must also stay together: merge groups that share an image hash
    by_hash = defaultdict(set)
    for r in clean:
        by_hash[r['sha256']].add(r['source'])
    sources = sorted(groups)
    random.Random(seed).shuffle(sources)
    # stratify by the source's majority (corrosion, severe) target so rare negatives land in every split
    strata = defaultdict(list)
    for s in sources:
        majority = Counter((r['corrosion'], r['severe']) for r in groups[s]).most_common(1)[0][0]
        strata[majority].append(s)
    split_of = {}
    for members in strata.values():
        n = len(members)
        for i, s in enumerate(members):
            split_of[s] = 'train' if i < round(.6 * n) else 'validation' if i < round(.8 * n) else 'test'
    for r in clean:
        r['split'] = split_of[r['source']]
    for r in records:
        r.setdefault('split', 'excluded_mosaic')
    shared = [h for h, s in by_hash.items() if len({split_of[x] for x in s}) > 1]
    if shared:
        raise SystemExit(f'{len(shared)} identical images would cross splits; refusing to write a leaky manifest')
    counts = {split: {'images': sum(r['split'] == split for r in clean), 'sources': len({r['source'] for r in clean if r['split'] == split}),
                      'corrosion_images': sum(r['corrosion'] for r in clean if r['split'] == split),
                      'no_corrosion_images': sum(not r['corrosion'] for r in clean if r['split'] == split),
                      'severe_images': sum(r['severe'] for r in clean if r['split'] == split)}
              for split in ('train', 'validation', 'test')}
    manifest = {'dataset': 'Roboflow Universe rust-detection-38s6e v1', 'license': 'CC BY 4.0', 'seed': seed,
                'excluded_mosaics': sum(r['mosaic'] for r in records), 'total_images': len(records), 'counts': counts,
                'targets': {'corrosion': 'any corrosion/rust label', 'severe': "worst grade is 'severe-corrosion'"},
                'limitations': 'Image-level tags from a public Roboflow project; noisy, not automotive-specific; mosaic detection is heuristic.',
                'records': records}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('total_images', 'excluded_mosaics', 'counts')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
